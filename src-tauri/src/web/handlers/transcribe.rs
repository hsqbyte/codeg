//! HTTP handlers for local speech-to-text.
//!
//! Both routes live on the token-protected `api` router (see `web::router`), so
//! a peer codeg calls them with `Authorization: Bearer <CODEG_TOKEN>`. They
//! share `commands::transcribe`'s `*_core` fns with the desktop Tauri commands.

use axum::Json;

use crate::app_error::AppCommandError;
use crate::commands::transcribe as core;
use crate::commands::transcribe::{SttCatalog, TranscribeRequest, TranscribeResult};

/// The model catalog + install state. No auth-sensitive data; lets a remote
/// client see which models the host has ready before sending audio.
pub async fn stt_catalog() -> Result<Json<SttCatalog>, AppCommandError> {
    Ok(Json(core::stt_catalog_core()))
}

/// Transcribe an uploaded 16 kHz mono WAV (base64) with the host's whisper
/// engine. Returns `DependencyMissing` if the host was built without
/// `stt-local`.
pub async fn transcribe(
    Json(req): Json<TranscribeRequest>,
) -> Result<Json<TranscribeResult>, AppCommandError> {
    Ok(Json(core::transcribe_core(req).await?))
}
