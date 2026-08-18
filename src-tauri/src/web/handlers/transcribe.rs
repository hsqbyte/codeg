//! HTTP handlers for local speech-to-text.
//!
//! Both routes live on the token-protected `api` router (see `web::router`), so
//! a peer codeg calls them with `Authorization: Bearer <CODEG_TOKEN>`. They
//! share `commands::transcribe`'s `*_core` fns with the desktop Tauri commands.

use std::sync::Arc;

use axum::{extract::Extension, Json};
use serde::Deserialize;

use crate::app_error::AppCommandError;
use crate::app_state::AppState;
use crate::commands::transcribe as core;
use crate::commands::transcribe::{
    DownloadModelRequest, SttCatalog, TranscribeRequest, TranscribeResult,
};

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

/// Download a model on the host, streaming progress over the shared event bus
/// (WS in server mode) — the frontend subscribes and filters by `taskId`.
pub async fn download_stt_model(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<DownloadModelRequest>,
) -> Result<Json<()>, AppCommandError> {
    core::download_stt_model_core(&state.emitter, req).await?;
    Ok(Json(()))
}

#[derive(Deserialize)]
pub struct CancelDownloadParams {
    pub task_id: String,
}

pub async fn cancel_stt_model_download(
    Json(params): Json<CancelDownloadParams>,
) -> Result<Json<()>, AppCommandError> {
    core::cancel_stt_model_download_core(&params.task_id);
    Ok(Json(()))
}

#[derive(Deserialize)]
pub struct DeleteModelParams {
    pub model_id: String,
}

pub async fn delete_stt_model(
    Json(params): Json<DeleteModelParams>,
) -> Result<Json<()>, AppCommandError> {
    core::delete_stt_model_core(&params.model_id)?;
    Ok(Json(()))
}
