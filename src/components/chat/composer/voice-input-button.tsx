"use client"

import { Loader2, Mic, Square } from "lucide-react"
import { useTranslations } from "next-intl"
import { useEffect } from "react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import { useVoiceInput } from "@/hooks/use-voice-input"

interface VoiceInputButtonProps {
  /** Receives the transcribed text (the composer inserts it at the cursor). */
  onInsert: (text: string) => void
  disabled?: boolean
}

/**
 * Mic button for the composer toolbar. Click to record, click again to stop and
 * transcribe (locally or via a remote codeg per Audio settings); the result is
 * handed to `onInsert`. Hidden entirely when the environment can't record
 * (non-secure context / no mic API) so it never shows a dead control.
 */
export function VoiceInputButton({
  onInsert,
  disabled,
}: VoiceInputButtonProps) {
  const t = useTranslations("Folder.chat.messageInput")
  const { status, available, toggle, error } = useVoiceInput(onInsert)

  useEffect(() => {
    if (error) toast.error(t("voiceInputError", { message: error }))
  }, [error, t])

  // A codeg served over plain http://LAN-IP can't reach the mic — don't render
  // a button that could never work.
  if (!available) return null

  const label =
    status === "recording"
      ? t("voiceInputStop")
      : status === "transcribing"
        ? t("voiceInputTranscribing")
        : t("voiceInput")

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      disabled={disabled || status === "transcribing"}
      onClick={toggle}
      title={label}
      aria-label={label}
      className={cn(
        status === "recording" && "text-red-500 hover:text-red-500"
      )}
    >
      {status === "recording" ? (
        <Square className="size-4 animate-pulse fill-current" />
      ) : status === "transcribing" ? (
        <Loader2 className="size-4 animate-spin" />
      ) : (
        <Mic className="size-4" />
      )}
    </Button>
  )
}
