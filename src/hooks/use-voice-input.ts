"use client"

import { useCallback, useEffect, useRef, useState } from "react"

import { transcribeAudio } from "@/lib/api"
import { toErrorMessage } from "@/lib/app-error"
import { canRecordAudio, startRecording } from "@/lib/audio-recorder"
import { useAudioInputPrefs } from "@/lib/audio-settings-prefs"
import type { TranscribeResult } from "@/lib/types"

export type VoiceInputStatus = "idle" | "recording" | "transcribing"

interface UseVoiceInput {
  status: VoiceInputStatus
  /** Mic recording is possible here (secure context + APIs present). */
  available: boolean
  /** idle → start recording; recording → stop + transcribe + emit text. */
  toggle: () => void
  /** Abort a recording without transcribing. */
  cancel: () => void
  /** Last error message, cleared on the next toggle. */
  error: string | null
}

/**
 * Drives click-to-toggle voice input: first click records from the mic, second
 * click stops and transcribes (locally via the bundled whisper engine, or by
 * POSTing to a remote codeg per the Audio settings), then hands the text to
 * `onText`. The caller inserts it (e.g. into the composer at the cursor).
 */
export function useVoiceInput(onText: (text: string) => void): UseVoiceInput {
  const prefs = useAudioInputPrefs()
  const [status, setStatus] = useState<VoiceInputStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<Awaited<ReturnType<typeof startRecording>> | null>(
    null
  )
  const [available, setAvailable] = useState(false)

  // Secure-context detection must run client-side (SSR export has no window).
  useEffect(() => {
    setAvailable(canRecordAudio())
  }, [])

  const transcribe = useCallback(
    async (audioBase64: string): Promise<string> => {
      const req = {
        audioBase64,
        modelId: prefs.modelId,
        language: prefs.language,
      }
      if (prefs.mode === "remote") {
        const base = prefs.remoteUrl.replace(/\/+$/, "")
        const res = await fetch(`${base}/api/transcribe`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${prefs.remoteToken}`,
          },
          body: JSON.stringify({ req }),
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const result = (await res.json()) as TranscribeResult
        return result.text
      }
      const result = await transcribeAudio(req)
      return result.text
    },
    [
      prefs.mode,
      prefs.modelId,
      prefs.language,
      prefs.remoteUrl,
      prefs.remoteToken,
    ]
  )

  const toggle = useCallback(() => {
    setError(null)
    if (status === "idle") {
      void (async () => {
        try {
          recorderRef.current = await startRecording()
          setStatus("recording")
        } catch (err) {
          setError(toErrorMessage(err))
          setStatus("idle")
        }
      })()
      return
    }
    if (status === "recording") {
      const recorder = recorderRef.current
      recorderRef.current = null
      if (!recorder) {
        setStatus("idle")
        return
      }
      setStatus("transcribing")
      void (async () => {
        try {
          const audioBase64 = await recorder.stop()
          const text = await transcribe(audioBase64)
          if (text.trim()) onText(text.trim())
        } catch (err) {
          setError(toErrorMessage(err))
        } finally {
          setStatus("idle")
        }
      })()
    }
  }, [status, transcribe, onText])

  const cancel = useCallback(() => {
    recorderRef.current?.cancel()
    recorderRef.current = null
    setStatus("idle")
  }, [])

  // Release the mic if the component unmounts mid-recording.
  useEffect(
    () => () => {
      recorderRef.current?.cancel()
      recorderRef.current = null
    },
    []
  )

  return { status, available, toggle, cancel, error }
}
