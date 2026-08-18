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

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::app_error::AppCommandError;
use crate::paths::codeg_stt_models_root;
use crate::web::event_bridge::{emit_event, EventEmitter};

/// Hard ceiling on a model download — `large-v3` is ~3.1 GB, so 5 GB leaves
/// headroom while rejecting a redirect to something absurd.
const MAX_MODEL_BYTES: u64 = 5 * 1024 * 1024 * 1024;

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
        // Always set the language — whisper-rs defaults to English when it is
        // left unset, which mis-transcribes Chinese speech as garbled English
        // (caught by the transcribe_smoke test). "auto" is whisper's own
        // detect token; an explicit code (e.g. "zh") pins detection.
        params.set_language(Some(&req.language));
        // Transcribe in the spoken language — never translate to English.
        params.set_translate(false);
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

// ── model download (streamed, cancellable) ──────────────────────────────────

const STT_MODEL_DOWNLOAD_EVENT: &str = "app://stt-model-download";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SttDownloadKind {
    Started,
    Progress,
    Completed,
    Failed,
}

/// Progress envelope streamed to the frontend during a model download. Keyed by
/// `task_id` so a component can filter to its own download (mirrors the ACP
/// agent-install event). `downloaded`/`total` drive a determinate progress bar;
/// `total` is 0 when the server sent no Content-Length.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SttDownloadEvent {
    pub task_id: String,
    pub kind: SttDownloadKind,
    pub model_id: String,
    pub downloaded: u64,
    pub total: u64,
    pub message: String,
}

fn emit_download_event(emitter: &EventEmitter, event: SttDownloadEvent) {
    emit_event(emitter, STT_MODEL_DOWNLOAD_EVENT, event);
}

/// Task ids whose download the user asked to cancel. The download loop checks
/// this each chunk and bails out, deleting the partial `.part` file. Process-
/// global (a `Mutex<HashSet>`) so the cancel command doesn't need AppState.
fn cancelled_downloads() -> &'static Mutex<HashSet<String>> {
    static CANCELLED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    CANCELLED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn take_cancel_flag(task_id: &str) -> bool {
    cancelled_downloads()
        .lock()
        .map(|mut set| set.remove(task_id))
        .unwrap_or(false)
}

pub fn cancel_stt_model_download_core(task_id: &str) {
    if let Ok(mut set) = cancelled_downloads().lock() {
        set.insert(task_id.to_string());
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadModelRequest {
    pub model_id: String,
    /// Client-minted id so the frontend can correlate progress events and
    /// cancel this specific download.
    pub task_id: String,
}

/// Download a model `.bin` into the STT models root, streaming progress events.
///
/// Writes to `ggml-<id>.bin.part` and renames to the final name only on
/// success, so an interrupted or cancelled download never leaves a truncated
/// file that `stt_catalog_core` would report as installed. Not feature-gated:
/// a headless `codeg-server` (with `stt-local`) can fetch models to serve peers.
pub async fn download_stt_model_core(
    emitter: &EventEmitter,
    req: DownloadModelRequest,
) -> Result<(), AppCommandError> {
    let model = find_model(&req.model_id)?;
    let final_path = model_file_path(&req.model_id)?;
    if final_path.exists() {
        emit_download_event(
            emitter,
            SttDownloadEvent {
                task_id: req.task_id.clone(),
                kind: SttDownloadKind::Completed,
                model_id: model.id.to_string(),
                downloaded: 0,
                total: 0,
                message: "already installed".to_string(),
            },
        );
        return Ok(());
    }

    let root = codeg_stt_models_root();
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(AppCommandError::io)?;
    let part_path = root.join(format!("ggml-{}.bin.part", model.id));

    // Clear any stale cancel flag from a previous attempt with this id.
    take_cancel_flag(&req.task_id);
    emit_download_event(
        emitter,
        SttDownloadEvent {
            task_id: req.task_id.clone(),
            kind: SttDownloadKind::Started,
            model_id: model.id.to_string(),
            downloaded: 0,
            total: model.size_bytes,
            message: "starting download".to_string(),
        },
    );

    let result =
        stream_to_file(emitter, model, &req.task_id, &part_path, &final_path).await;

    match &result {
        Ok(()) => emit_download_event(
            emitter,
            SttDownloadEvent {
                task_id: req.task_id.clone(),
                kind: SttDownloadKind::Completed,
                model_id: model.id.to_string(),
                downloaded: model.size_bytes,
                total: model.size_bytes,
                message: "done".to_string(),
            },
        ),
        Err(e) => {
            // Best-effort cleanup of the partial file on any failure/cancel.
            let _ = tokio::fs::remove_file(&part_path).await;
            emit_download_event(
                emitter,
                SttDownloadEvent {
                    task_id: req.task_id.clone(),
                    kind: SttDownloadKind::Failed,
                    model_id: model.id.to_string(),
                    downloaded: 0,
                    total: model.size_bytes,
                    message: e.message.clone(),
                },
            );
        }
    }

    result
}

async fn stream_to_file(
    emitter: &EventEmitter,
    model: &SttModel,
    task_id: &str,
    part_path: &std::path::Path,
    final_path: &std::path::Path,
) -> Result<(), AppCommandError> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let url = model_download_url(model.id)?;
    let response = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .map_err(|e| AppCommandError::network(format!("download request failed: {e}")))?;
    if !response.status().is_success() {
        return Err(AppCommandError::network(format!(
            "download returned status {}",
            response.status()
        )));
    }
    let total = response.content_length().unwrap_or(model.size_bytes);
    if total > MAX_MODEL_BYTES {
        return Err(AppCommandError::invalid_input(format!(
            "model download is unexpectedly large ({total} bytes)"
        )));
    }

    let mut file = tokio::fs::File::create(part_path)
        .await
        .map_err(AppCommandError::io)?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    // Emit at most ~every 8 MB so a 3 GB model yields a smooth bar without
    // flooding the event bus.
    let mut last_emit: u64 = 0;
    const EMIT_STEP: u64 = 8 * 1024 * 1024;

    while let Some(chunk) = stream.next().await {
        if take_cancel_flag(task_id) {
            drop(file);
            return Err(AppCommandError::task_execution_failed("download cancelled"));
        }
        let chunk =
            chunk.map_err(|e| AppCommandError::network(format!("download interrupted: {e}")))?;
        downloaded += chunk.len() as u64;
        if downloaded > MAX_MODEL_BYTES {
            drop(file);
            return Err(AppCommandError::invalid_input(
                "model download exceeded the maximum allowed size",
            ));
        }
        file.write_all(&chunk).await.map_err(AppCommandError::io)?;
        if downloaded - last_emit >= EMIT_STEP {
            last_emit = downloaded;
            emit_download_event(
                emitter,
                SttDownloadEvent {
                    task_id: task_id.to_string(),
                    kind: SttDownloadKind::Progress,
                    model_id: model.id.to_string(),
                    downloaded,
                    total,
                    message: String::new(),
                },
            );
        }
    }
    file.flush().await.map_err(AppCommandError::io)?;
    drop(file);

    // Atomic-ish publish: rename the completed .part to the real name.
    tokio::fs::rename(part_path, final_path)
        .await
        .map_err(AppCommandError::io)?;
    Ok(())
}

/// Delete a downloaded model file. No-op (Ok) if it isn't present.
pub fn delete_stt_model_core(model_id: &str) -> Result<(), AppCommandError> {
    let path = model_file_path(model_id)?;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(AppCommandError::io(e)),
    }
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

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn download_stt_model(
    req: DownloadModelRequest,
    app: tauri::AppHandle,
) -> Result<(), AppCommandError> {
    let emitter = EventEmitter::Tauri(app);
    download_stt_model_core(&emitter, req).await
}

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn cancel_stt_model_download(task_id: String) -> Result<(), AppCommandError> {
    cancel_stt_model_download_core(&task_id);
    Ok(())
}

#[cfg(feature = "tauri-runtime")]
#[cfg_attr(feature = "tauri-runtime", tauri::command)]
pub async fn delete_stt_model(model_id: String) -> Result<(), AppCommandError> {
    delete_stt_model_core(&model_id)
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

    /// End-to-end smoke test of `transcribe_core`'s new-code path (base64 →
    /// hound WAV decode → whisper) against a real model + WAV. `#[ignore]`d
    /// because it needs a downloaded model; run with:
    ///   STT_TEST_MODEL=<ggml-*.bin> STT_TEST_WAV=<16k-mono.wav> \
    ///   CODEG_HOME=<tmp> cargo test --features test-utils \
    ///     transcribe_smoke -- --ignored --nocapture
    /// (CODEG_HOME must contain stt-models/ggml-<id>.bin — the harness copies
    /// STT_TEST_MODEL there.)
    #[cfg(feature = "stt-local")]
    #[tokio::test]
    #[ignore]
    async fn transcribe_smoke_transcribes_real_audio() {
        use base64::Engine;

        let model_src = std::env::var("STT_TEST_MODEL").expect("set STT_TEST_MODEL");
        let wav_path = std::env::var("STT_TEST_WAV").expect("set STT_TEST_WAV");

        // Stage the model into the models root under whatever id the file name
        // implies (default to `base`).
        let root = codeg_stt_models_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::copy(&model_src, root.join("ggml-base.bin")).unwrap();

        let wav = std::fs::read(&wav_path).unwrap();
        let audio_base64 = base64::engine::general_purpose::STANDARD.encode(&wav);

        let res = transcribe_core(TranscribeRequest {
            audio_base64,
            model_id: "base".to_string(),
            language: "auto".to_string(),
        })
        .await
        .expect("transcription should succeed");

        eprintln!("transcript: {:?} ({}ms)", res.text, res.elapsed_ms);
        assert!(!res.text.is_empty(), "expected non-empty transcript");
    }
}
