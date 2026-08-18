"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { useTranslations } from "next-intl"
import { Check, Download, Loader2, Mic, Trash2, X } from "lucide-react"

import {
  SettingCard,
  SettingNote,
  SettingRow,
} from "@/components/shared/setting-card"
import { SettingsSection } from "@/components/shared/settings-section"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { toast } from "sonner"
import {
  cancelSttModelDownload,
  deleteSttModel,
  downloadSttModel,
  getSttCatalog,
} from "@/lib/api"
import { randomUUID } from "@/lib/utils"
import { toErrorMessage } from "@/lib/app-error"
import {
  STT_LANGUAGES,
  useAudioInputPrefs,
  updateAudioInputPrefs,
} from "@/lib/audio-settings-prefs"
import { useSttModelDownload } from "@/hooks/use-stt-model-download"
import type { SttCatalog } from "@/lib/types"

function formatBytes(n: number): string {
  if (n <= 0) return "0 MB"
  const gb = n / 1024 ** 3
  if (gb >= 1) return `${gb.toFixed(1)} GB`
  return `${Math.round(n / 1024 ** 2)} MB`
}

export function AudioSettings() {
  const t = useTranslations("AudioSettings")
  const prefs = useAudioInputPrefs()
  const [catalog, setCatalog] = useState<SttCatalog | null>(null)
  const [testing, setTesting] = useState(false)
  const download = useSttModelDownload()
  const currentTaskId = useRef<string | null>(null)

  const refreshCatalog = useCallback(async () => {
    try {
      setCatalog(await getSttCatalog())
    } catch {
      // Non-fatal: the panel still renders from the static model list on the
      // next successful fetch; a hard failure here just means no install state.
    }
  }, [])

  useEffect(() => {
    void refreshCatalog()
  }, [refreshCatalog])

  // Refetch install state whenever a download finishes so the button flips to
  // "installed" without a manual reload.
  const prevStatus = useRef(download.status)
  useEffect(() => {
    if (prevStatus.current === "running" && download.status === "success") {
      void refreshCatalog()
    }
    prevStatus.current = download.status
  }, [download.status, refreshCatalog])

  useEffect(() => () => download.reset(), [download])

  const handleDownload = useCallback(
    async (modelId: string) => {
      const taskId = randomUUID()
      currentTaskId.current = taskId
      await download.start(taskId, modelId)
      try {
        await downloadSttModel({ modelId, taskId })
      } catch (err) {
        toast.error(t("downloadFailed", { message: toErrorMessage(err) }))
      }
    },
    [download, t]
  )

  const handleCancel = useCallback(async () => {
    // Signal the backend to abort mid-stream (it deletes the partial file on
    // the next chunk check), then locally reset the progress UI.
    if (currentTaskId.current) {
      try {
        await cancelSttModelDownload(currentTaskId.current)
      } catch {
        /* best-effort */
      }
    }
    download.reset()
  }, [download])

  const handleDelete = useCallback(
    async (modelId: string) => {
      try {
        await deleteSttModel(modelId)
        await refreshCatalog()
      } catch (err) {
        toast.error(toErrorMessage(err))
      }
    },
    [refreshCatalog]
  )

  const handleTestRemote = useCallback(async () => {
    if (!prefs.remoteUrl) return
    setTesting(true)
    try {
      const base = prefs.remoteUrl.replace(/\/+$/, "")
      const res = await fetch(`${base}/api/stt_catalog`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${prefs.remoteToken}`,
        },
        body: "{}",
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const remote = (await res.json()) as SttCatalog
      const installed = remote.models.filter((m) => m.installed).length
      toast.success(t("remoteOk", { count: installed }))
    } catch (err) {
      toast.error(t("remoteFailed", { message: toErrorMessage(err) }))
    } finally {
      setTesting(false)
    }
  }, [prefs.remoteUrl, prefs.remoteToken, t])

  const localUnavailable = catalog !== null && !catalog.localAvailable
  const pct =
    download.total > 0
      ? Math.min(100, Math.round((download.downloaded / download.total) * 100))
      : 0

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-2xl space-y-6 p-4">
        <SettingsSection
          icon={Mic}
          title={t("title")}
          description={t("subtitle")}
        >
          <Tabs
            value={prefs.mode}
            onValueChange={(v) =>
              updateAudioInputPrefs({
                mode: v === "remote" ? "remote" : "local",
              })
            }
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="local">{t("modeLocal")}</TabsTrigger>
              <TabsTrigger value="remote">{t("modeRemote")}</TabsTrigger>
            </TabsList>
          </Tabs>

          {/* Language applies to both modes. */}
          <SettingCard>
            <SettingRow title={t("language")} description={t("languageHint")}>
              <Select
                value={prefs.language}
                onValueChange={(v) => updateAudioInputPrefs({ language: v })}
              >
                <SelectTrigger className="w-36">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {STT_LANGUAGES.map((lang) => (
                    <SelectItem key={lang} value={lang}>
                      {t(`lang_${lang}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </SettingRow>
          </SettingCard>

          {prefs.mode === "local" ? (
            <>
              {localUnavailable ? (
                <SettingNote>{t("localUnavailable")}</SettingNote>
              ) : (
                <SettingNote>{t("localHint")}</SettingNote>
              )}
              <SettingCard>
                {(catalog?.models ?? []).map((m) => {
                  const isActive = prefs.modelId === m.id
                  const isDownloading =
                    download.status === "running" && download.modelId === m.id
                  return (
                    <SettingRow
                      key={m.id}
                      title={
                        <span className="flex items-center gap-2">
                          {m.label}
                          <span className="text-xs text-muted-foreground">
                            {formatBytes(m.sizeBytes)}
                          </span>
                          {isActive ? (
                            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                              {t("active")}
                            </span>
                          ) : null}
                        </span>
                      }
                      description={
                        isDownloading ? (
                          <span className="flex items-center gap-2">
                            <Progress value={pct} className="h-1.5 w-40" />
                            <span className="text-xs tabular-nums">
                              {pct}% · {formatBytes(download.downloaded)}/
                              {formatBytes(download.total || m.sizeBytes)}
                            </span>
                          </span>
                        ) : (
                          m.note
                        )
                      }
                    >
                      <div className="flex items-center gap-1">
                        {isDownloading ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={handleCancel}
                          >
                            <X className="size-4" />
                            {t("cancel")}
                          </Button>
                        ) : m.installed ? (
                          <>
                            {!isActive ? (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() =>
                                  updateAudioInputPrefs({ modelId: m.id })
                                }
                              >
                                {t("use")}
                              </Button>
                            ) : (
                              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                <Check className="size-3.5" /> {t("installed")}
                              </span>
                            )}
                            <Button
                              size="icon"
                              variant="ghost"
                              className="size-8"
                              title={t("delete")}
                              onClick={() => handleDelete(m.id)}
                            >
                              <Trash2 className="size-4" />
                            </Button>
                          </>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={download.status === "running"}
                            onClick={() => handleDownload(m.id)}
                          >
                            {download.status === "running" ? (
                              <Loader2 className="size-4 animate-spin" />
                            ) : (
                              <Download className="size-4" />
                            )}
                            {t("download")}
                          </Button>
                        )}
                      </div>
                    </SettingRow>
                  )
                })}
              </SettingCard>
            </>
          ) : (
            <>
              <SettingNote>{t("remoteHint")}</SettingNote>
              <SettingCard>
                <div className="space-y-3 p-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="stt-remote-url">{t("remoteUrl")}</Label>
                    <Input
                      id="stt-remote-url"
                      placeholder="https://mini.local:8788"
                      value={prefs.remoteUrl}
                      onChange={(e) =>
                        updateAudioInputPrefs({ remoteUrl: e.target.value })
                      }
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="stt-remote-token">{t("remoteToken")}</Label>
                    <Input
                      id="stt-remote-token"
                      type="password"
                      value={prefs.remoteToken}
                      onChange={(e) =>
                        updateAudioInputPrefs({ remoteToken: e.target.value })
                      }
                    />
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!prefs.remoteUrl || testing}
                    onClick={handleTestRemote}
                  >
                    {testing ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : null}
                    {t("testConnection")}
                  </Button>
                </div>
              </SettingCard>
            </>
          )}
        </SettingsSection>
      </div>
    </ScrollArea>
  )
}
