import { useCallback, useRef, useState } from "react"
import { subscribe } from "@/lib/platform"
import type { SttDownloadEvent, SttDownloadKind } from "@/lib/types"

const STT_MODEL_DOWNLOAD_EVENT = "app://stt-model-download"

export type SttDownloadStatus = "idle" | "running" | "success" | "failed"

interface SttDownloadState {
  status: SttDownloadStatus
  /** Which model this run is downloading (null when idle). */
  modelId: string | null
  downloaded: number
  total: number
  error: string | null
}

const IDLE: SttDownloadState = {
  status: "idle",
  modelId: null,
  downloaded: 0,
  total: 0,
  error: null,
}

/**
 * Drives a whisper-model download and its `app://stt-model-download` progress
 * stream. Mirrors `use-officecli-install-stream` (including the `cancelledRef`
 * teardown guard) but tracks byte progress for a determinate bar rather than a
 * log list. Filters events by `taskId` so two panels never cross streams.
 */
export function useSttModelDownload() {
  const [state, setState] = useState<SttDownloadState>(IDLE)
  const unsubRef = useRef<(() => void) | null>(null)
  const cancelledRef = useRef(false)

  const start = useCallback(async (taskId: string, modelId: string) => {
    cancelledRef.current = false
    setState({ ...IDLE, status: "running", modelId })

    unsubRef.current?.()

    const unsub = await subscribe<SttDownloadEvent>(
      STT_MODEL_DOWNLOAD_EVENT,
      (event) => {
        if (event.taskId !== taskId) return

        switch (event.kind as SttDownloadKind) {
          case "started":
            setState((prev) => ({
              ...prev,
              status: "running",
              total: event.total || prev.total,
            }))
            break
          case "progress":
            setState((prev) => ({
              ...prev,
              status: "running",
              downloaded: event.downloaded,
              total: event.total || prev.total,
            }))
            break
          case "completed":
            setState((prev) => ({
              ...prev,
              status: "success",
              downloaded: prev.total,
            }))
            unsubRef.current?.()
            break
          case "failed":
            setState((prev) => ({
              ...prev,
              status: "failed",
              error: event.message,
            }))
            unsubRef.current?.()
            break
        }
      }
    )

    if (cancelledRef.current) {
      unsub()
      return
    }
    unsubRef.current = unsub
  }, [])

  const reset = useCallback(() => {
    cancelledRef.current = true
    unsubRef.current?.()
    unsubRef.current = null
    setState(IDLE)
  }, [])

  return { ...state, start, reset }
}
