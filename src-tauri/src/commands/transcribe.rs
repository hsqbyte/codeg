//! Local speech-to-text (whisper) commands.
//!
//! The `*_core` fns are mode-agnostic (plain values, no `tauri::State`) so the
//! `#[tauri::command]` wrappers and the Axum handlers share one code path — a
//! desktop app transcribing locally and a peer codeg calling `/api/transcribe`
//! run the exact same `transcribe_core`.
//!
//! All whisper.cpp usage is gated behind `#[cfg(feature = "stt-local")]`. With
//! the feature off (the default for `codeg-server` / `codeg-mcp`), the catalog
//! and installed-model helpers still compile, and `transcribe_core` returns a
//! `DependencyMissing` error instead of pulling whisper.cpp into the build.

use serde::{Deserialize, Serialize};

use crate::app_error::AppCommandError;
use crate::paths::codeg_stt_models_root;

/// One selectable whisper model. `id` doubles as the on-disk filename stem
/// (`ggml-<id>.bin`) and the download-URL stem on the ggml HF mirror.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SttModel {
    /// Stable identifier, e.g. `"large-v3-turbo"`.
    pub id: &'static str,
    /// Human label for the settings UI, e.g. `"Large v3 Turbo"`.
    pub label: &'static str,
    /// Approximate download size in bytes (for the progress bar's total when
    /// the server does not send a Content-Length).
    pub size_bytes: u64,
    /// One-line tradeoff note shown under the label.
    pub note: &'static str,
}

/// The models offered in Audio settings, smallest → largest. `large-v3-turbo`
/// is the recommended default: near-`large-v3` quality at roughly half the
/// size and much faster, which on Apple-silicon Metal is effectively instant.
pub const STT_MODELS: &[SttModel] = &[
    SttModel {
        id: "base",
        label: "Base",
        size_bytes: 147_951_465,
        note: "最小、最快，中英混杂够用，适合先试手感",
    },
    SttModel {
        id: "medium",
        label: "Medium",
        size_bytes: 1_533_763_059,
        note: "质量与体积折中",
    },
    SttModel {
        id: "large-v3-turbo",
        label: "Large v3 Turbo",
        size_bytes: 1_624_555_275,
        note: "推荐：接近 large-v3 的质量，速度快、体积小一半",
    },
    SttModel {
        id: "large-v3",
        label: "Large v3",
        size_bytes: 3_095_033_483,
        note: "质量最高，体积最大、稍慢",
    },
];

/// The ggml HF mirror the download command pulls model files from.
pub const STT_MODEL_URL_BASE: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Look up a model by id, rejecting anything not in [`STT_MODELS`] so a
/// caller-supplied id can never escape the models root or hit an arbitrary URL.
pub fn find_model(id: &str) -> Result<&'static SttModel, AppCommandError> {
    STT_MODELS
        .iter()
        .find(|m| m.id == id)
        .ok_or_else(|| AppCommandError::invalid_input(format!("unknown STT model: {id}")))
}

/// Absolute path of a model's `.bin` inside the STT models root.
pub fn model_file_path(id: &str) -> Result<std::path::PathBuf, AppCommandError> {
    let model = find_model(id)?;
    Ok(codeg_stt_models_root().join(format!("ggml-{}.bin", model.id)))
}

/// The download URL for a model's `.bin`.
pub fn model_download_url(id: &str) -> Result<String, AppCommandError> {
    let model = find_model(id)?;
    Ok(format!("{STT_MODEL_URL_BASE}/ggml-{}.bin", model.id))
}

/// The catalog plus which models are already downloaded — drives the settings
/// UI (show a "download" vs "installed" state per model).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SttCatalog {
    pub models: Vec<SttCatalogEntry>,
    /// Whether this binary was built with the `stt-local` feature. When false,
    /// the frontend hides local mode and steers the user to remote mode.
    pub local_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SttCatalogEntry {
    pub id: String,
    pub label: String,
    pub size_bytes: u64,
    pub note: String,
    /// The `.bin` exists in the models root (rough check: present and
    /// non-empty; a partial download is cleaned up by the download command).
    pub installed: bool,
}

pub fn stt_catalog_core() -> SttCatalog {
    let models = STT_MODELS
        .iter()
        .map(|m| {
            let installed = model_file_path(m.id)
                .ok()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|meta| meta.len() > 0)
                .unwrap_or(false);
            SttCatalogEntry {
                id: m.id.to_string(),
                label: m.label.to_string(),
                size_bytes: m.size_bytes,
                note: m.note.to_string(),
                installed,
            }
        })
        .collect();
    SttCatalog {
        models,
        local_available: cfg!(feature = "stt-local"),
    }
}

/// A transcription request. `audio_base64` is a 16 kHz mono 16-bit PCM WAV
/// (what the frontend records); `language` is a whisper language code such as
/// `"zh"`, `"en"`, or `"auto"` to detect.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeRequest {
    pub audio_base64: String,
    pub model_id: String,
    #[serde(default = "default_language")]
    pub language: String,
}

fn default_language() -> String {
    "auto".to_string()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeResult {
    pub text: String,
    /// Wall-clock transcription time in milliseconds (excludes model load).
    pub elapsed_ms: u64,
    /// Decoded audio duration in milliseconds.
    pub audio_ms: u64,
}

/// Decode the base64 WAV into 16 kHz mono f32 PCM. Shared by both builds so the
/// non-feature build still validates input the same way (and its error message
/// is identical up to the final "feature unavailable").
#[cfg(feature = "stt-local")]
fn decode_wav_16k_mono(audio_base64: &str) -> Result<Vec<f32>, AppCommandError> {
    use base64::Engine;
    use std::io::Cursor;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_base64.as_bytes())
        .map_err(|e| AppCommandError::invalid_input(format!("invalid audio base64: {e}")))?;

    let reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| AppCommandError::invalid_input(format!("not a valid WAV: {e}")))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err(AppCommandError::invalid_input(format!(
            "expected 16kHz mono WAV, got {}Hz {}ch",
            spec.sample_rate, spec.channels
        )));
    }

    // The frontend records 16-bit integer PCM; convert to the f32 [-1, 1]
    // range whisper wants.
    let samples: Result<Vec<f32>, _> = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect();
    samples.map_err(|e| AppCommandError::invalid_input(format!("corrupt WAV samples: {e}")))
}

/// Transcribe recorded audio locally with whisper.cpp.
///
/// Blocking whisper work runs on a `spawn_blocking` thread so it never stalls
/// the async runtime (a `large-v3` pass is CPU/GPU-heavy). Returns
/// `DependencyMissing` when built without `stt-local` — the frontend treats
/// that as "this instance can't transcribe locally, use remote mode".
#[cfg(feature = "stt-local")]
pub async fn transcribe_core(
    req: TranscribeRequest,
) -> Result<TranscribeResult, AppCommandError> {
    use whisper_rs::{
        FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    };

    let model_path = model_file_path(&req.model_id)?;
    if !model_path.exists() {
        return Err(AppCommandError::configuration_missing(format!(
            "STT model '{}' is not downloaded yet",
            req.model_id
        )));
    }

    let samples = decode_wav_16k_mono(&req.audio_base64)?;
    let audio_ms = (samples.len() as u64) * 1000 / 16_000;

    let text = tokio::task::spawn_blocking(move || -> Result<(String, u64), AppCommandError> {
        let model_path_str = model_path.to_string_lossy().to_string();
        let ctx = WhisperContext::new_with_params(
            &model_path_str,
            WhisperContextParameters::default(),
        )
        .map_err(|e| {
            AppCommandError::task_execution_failed(format!("failed to load STT model: {e}"))
        })?;
        let mut state = ctx.create_state().map_err(|e| {
            AppCommandError::task_execution_failed(format!("whisper state init failed: {e}"))
        })?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // "auto" → let whisper detect; whisper-rs takes None for auto-detect.
        if req.language != "auto" {
            params.set_language(Some(&req.language));
        }
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);

        let started = std::time::Instant::now();
        state.full(params, &samples).map_err(|e| {
            AppCommandError::task_execution_failed(format!("transcription failed: {e}"))
        })?;
        let elapsed_ms = started.elapsed().as_millis() as u64;

        let n = state.full_n_segments().map_err(|e| {
            AppCommandError::task_execution_failed(format!("segment count failed: {e}"))
        })?;
        let mut text = String::new();
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                text.push_str(&seg);
            }
        }
        Ok((text.trim().to_string(), elapsed_ms))
    })
    .await
    .map_err(|e| AppCommandError::task_execution_failed(format!("STT task panicked: {e}")))??;

    Ok(TranscribeResult {
        text: text.0,
        elapsed_ms: text.1,
        audio_ms,
    })
}

/// Feature-off stub: keeps `codeg-server` / `codeg-mcp` compiling without
/// whisper.cpp. Returns the same shape the frontend already handles.
#[cfg(not(feature = "stt-local"))]
pub async fn transcribe_core(
    _req: TranscribeRequest,
) -> Result<TranscribeResult, AppCommandError> {
    Err(AppCommandError::dependency_missing(
        "this codeg build has no local speech-to-text engine (feature `stt-local` is off); \
         use remote transcription mode instead",
    ))
}

// ── Tauri command wrappers (desktop-only) ───────────────────────────────────

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn stt_catalog() -> Result<SttCatalog, AppCommandError> {
    Ok(stt_catalog_core())
}

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn transcribe(req: TranscribeRequest) -> Result<TranscribeResult, AppCommandError> {
    transcribe_core(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_all_models_with_local_flag_matching_build() {
        let cat = stt_catalog_core();
        assert_eq!(cat.models.len(), STT_MODELS.len());
        assert_eq!(cat.local_available, cfg!(feature = "stt-local"));
    }

    #[test]
    fn find_model_rejects_unknown_id() {
        assert!(find_model("../../etc/passwd").is_err());
        assert!(find_model("large-v3-turbo").is_ok());
    }

    #[test]
    fn model_path_stays_inside_models_root() {
        let path = model_file_path("large-v3-turbo").unwrap();
        assert!(path.starts_with(codeg_stt_models_root()));
        assert!(path.ends_with("ggml-large-v3-turbo.bin"));
    }

    #[test]
    fn download_url_is_on_the_ggml_mirror() {
        let url = model_download_url("base").unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
        );
    }
}
