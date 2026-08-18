"use client"

/**
 * Microphone → 16 kHz mono 16-bit PCM WAV recorder for voice input.
 *
 * Whisper wants 16 kHz mono f32; the backend decodes a 16-bit PCM WAV. Rather
 * than record Opus/webm via MediaRecorder and decode server-side, we open an
 * AudioContext pinned to 16 kHz and capture raw PCM through a
 * ScriptProcessorNode, then encode a WAV in the browser — one format, no
 * server-side transcoding.
 *
 * ScriptProcessorNode is deprecated in favour of AudioWorklet but is supported
 * everywhere (including older Safari/WebViews) and needs no separate worklet
 * module to load — a good fit for the Tauri webview and self-hosted web client.
 */

/** True only in a secure context with mic APIs — getUserMedia requires HTTPS
 *  or localhost, so a codeg served over plain http://LAN-IP can't record. */
export function canRecordAudio(): boolean {
  if (typeof window === "undefined") return false
  return (
    Boolean(window.isSecureContext) &&
    typeof navigator !== "undefined" &&
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    typeof (
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: unknown }).webkitAudioContext
    ) !== "undefined"
  )
}

const TARGET_RATE = 16_000

interface ActiveRecording {
  /** Stop capture and return the recorded audio as a base64 16 kHz mono WAV. */
  stop: () => Promise<string>
  /** Abort without producing audio (releases the mic). */
  cancel: () => void
}

export async function startRecording(): Promise<ActiveRecording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext
  // Ask for 16 kHz; browsers that honour it resample for us. Where they don't,
  // we down-sample from the actual rate at encode time.
  const ctx = new Ctx({ sampleRate: TARGET_RATE })
  const source = ctx.createMediaStreamSource(stream)
  const processor = ctx.createScriptProcessor(4096, 1, 1)
  const chunks: Float32Array[] = []

  processor.onaudioprocess = (e) => {
    // Copy — the event buffer is reused across callbacks.
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
  }
  source.connect(processor)
  processor.connect(ctx.destination)

  const teardown = () => {
    processor.disconnect()
    source.disconnect()
    stream.getTracks().forEach((t) => t.stop())
    void ctx.close()
  }

  return {
    stop: async () => {
      const inputRate = ctx.sampleRate
      teardown()
      const samples = downsampleTo16k(mergeChunks(chunks), inputRate)
      return encodeWavBase64(samples)
    },
    cancel: teardown,
  }
}

function mergeChunks(chunks: Float32Array[]): Float32Array {
  let length = 0
  for (const c of chunks) length += c.length
  const out = new Float32Array(length)
  let offset = 0
  for (const c of chunks) {
    out.set(c, offset)
    offset += c.length
  }
  return out
}

/** Linear-interpolation down-sample. A no-op when the context already gave us
 *  16 kHz (the common case); the fallback covers browsers that ignore the
 *  requested rate and hand back 44.1/48 kHz. */
function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_RATE || input.length === 0) return input
  const ratio = inputRate / TARGET_RATE
  const outLength = Math.floor(input.length / ratio)
  const out = new Float32Array(outLength)
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio
    const i0 = Math.floor(pos)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const frac = pos - i0
    out[i] = input[i0] * (1 - frac) + input[i1] * frac
  }
  return out
}

/** Encode 16 kHz mono f32 samples as a 16-bit PCM WAV, base64-encoded. */
function encodeWavBase64(samples: Float32Array): string {
  const dataBytes = samples.length * 2
  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++)
      view.setUint8(offset + i, s.charCodeAt(i))
  }

  writeStr(0, "RIFF")
  view.setUint32(4, 36 + dataBytes, true)
  writeStr(8, "WAVE")
  writeStr(12, "fmt ")
  view.setUint32(16, 16, true) // fmt chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, TARGET_RATE, true)
  view.setUint32(28, TARGET_RATE * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeStr(36, "data")
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }

  // Base64 the bytes in chunks to avoid a String.fromCharCode stack overflow.
  const bytes = new Uint8Array(buffer)
  let binary = ""
  const CHUNK = 0x8000
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
  }
  return btoa(binary)
}
