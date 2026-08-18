"use client"

/**
 * Voice-input (speech-to-text) preferences: whether to transcribe on this
 * machine (`local`, using the bundled whisper engine + a downloaded model) or
 * hand audio to another codeg over its API (`remote`), which model to use, the
 * spoken language, and the remote endpoint.
 *
 * Stored in localStorage rather than the backend for the same reason as
 * `notification-sound-prefs.ts`: voice input is a per-device concern — a phone
 * browser attached to the same server picks its own mode/endpoint independent
 * of the desktop. Same reactive shape (a custom event for this window plus the
 * native `storage` event for other windows/tabs), so changing the mode in the
 * Settings window takes effect in the composer's mic button immediately.
 *
 * The remote token lives here in plaintext. That is acceptable for this
 * per-device, user-controlled setting (it is the shared CODEG_TOKEN of a codeg
 * the user themselves runs); it is never sent anywhere except as the `Bearer`
 * auth to that endpoint.
 */

import { useSyncExternalStore } from "react"

const PREFS_KEY = "settings:audio-input:v1"
const PREFS_EVENT = "codeg:audio-input-changed"

export type SttMode = "local" | "remote"

/** Language options offered in the panel. `auto` lets whisper detect. */
export const STT_LANGUAGES = ["auto", "zh", "en", "ja", "ko"] as const
export type SttLanguage = (typeof STT_LANGUAGES)[number]

export interface AudioInputPrefs {
  mode: SttMode
  /** Model id from the backend catalog (e.g. "large-v3-turbo"). */
  modelId: string
  /** Whisper language code or "auto". */
  language: string
  /** Remote codeg base URL, e.g. "https://mini.local:8788" (no trailing /). */
  remoteUrl: string
  /** Remote codeg's CODEG_TOKEN. */
  remoteToken: string
}

export const DEFAULT_AUDIO_INPUT_PREFS: AudioInputPrefs = {
  mode: "local",
  modelId: "large-v3-turbo",
  language: "auto",
  remoteUrl: "",
  remoteToken: "",
}

function str(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback
}

/** Merge a stored blob over the defaults, field by field, so a partial or
 *  older write degrades per-field instead of discarding everything. */
export function parseAudioInputPrefs(raw: unknown): AudioInputPrefs {
  const d = DEFAULT_AUDIO_INPUT_PREFS
  if (!raw || typeof raw !== "object") return { ...d }
  const s = raw as Record<string, unknown>
  return {
    mode: s.mode === "remote" ? "remote" : "local",
    modelId: str(s.modelId, d.modelId),
    language: str(s.language, d.language),
    remoteUrl: str(s.remoteUrl, d.remoteUrl),
    remoteToken: str(s.remoteToken, d.remoteToken),
  }
}

export function loadAudioInputPrefs(): AudioInputPrefs {
  if (typeof window === "undefined") return { ...DEFAULT_AUDIO_INPUT_PREFS }
  try {
    const raw = localStorage.getItem(PREFS_KEY)
    if (!raw) return { ...DEFAULT_AUDIO_INPUT_PREFS }
    return parseAudioInputPrefs(JSON.parse(raw))
  } catch {
    return { ...DEFAULT_AUDIO_INPUT_PREFS }
  }
}

export function saveAudioInputPrefs(prefs: AudioInputPrefs): void {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs))
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent(PREFS_EVENT, { detail: prefs }))
}

// ── Shared snapshot (memoized per window, invalidated on any write) ──

let snapshot: AudioInputPrefs | null = null
const listeners = new Set<() => void>()
let windowBound = false

function bindWindow(): void {
  if (windowBound || typeof window === "undefined") return
  windowBound = true
  const invalidate = () => {
    snapshot = null
    for (const listener of listeners) listener()
  }
  window.addEventListener(PREFS_EVENT, invalidate)
  window.addEventListener("storage", invalidate)
}

export function getAudioInputPrefs(): AudioInputPrefs {
  bindWindow()
  if (typeof window === "undefined") return DEFAULT_AUDIO_INPUT_PREFS
  snapshot ??= loadAudioInputPrefs()
  return snapshot
}

export function subscribeAudioInputPrefs(onChange: () => void): () => void {
  bindWindow()
  listeners.add(onChange)
  return () => {
    listeners.delete(onChange)
  }
}

function getServerAudioInputPrefs(): AudioInputPrefs {
  return DEFAULT_AUDIO_INPUT_PREFS
}

/** Reactive read of the preferences; live across windows. */
export function useAudioInputPrefs(): AudioInputPrefs {
  return useSyncExternalStore(
    subscribeAudioInputPrefs,
    getAudioInputPrefs,
    getServerAudioInputPrefs
  )
}

/** Patch a subset of fields and persist. */
export function updateAudioInputPrefs(patch: Partial<AudioInputPrefs>): void {
  saveAudioInputPrefs({ ...getAudioInputPrefs(), ...patch })
}
