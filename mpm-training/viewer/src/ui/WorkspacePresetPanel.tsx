import { useRef, useState } from "react"
import { LAST_PRESET_KEY, PRESETS_KEY, readWorkspacePresets, type WorkspacePreset, type WorkspacePresetValue } from "../performance/workspacePresets"

export function WorkspacePresetPanel({ capture, onLoad }: {
  capture: () => WorkspacePresetValue | null
  onLoad: (value: WorkspacePresetValue) => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [mode, setMode] = useState<"save" | "load">("load")
  const [name, setName] = useState("")
  const [presets, setPresets] = useState(readWorkspacePresets)
  const [error, setError] = useState("")
  const [status, setStatus] = useState("")
  const open = (next: "save" | "load") => { setMode(next); setError(""); setPresets(readWorkspacePresets()); dialog.current?.showModal() }
  const save = () => {
    const value = capture(); if (!value) { setError("The simulation is not ready to save yet."); return }
    const preset: WorkspacePreset = { version: 1, id: crypto.randomUUID(), name: name.trim() || `Preset ${presets.length + 1}`, savedAt: new Date().toISOString(), value }
    try {
      localStorage.setItem(PRESETS_KEY, JSON.stringify([...presets, preset]))
      localStorage.setItem(LAST_PRESET_KEY, preset.id)
      setStatus(`Saved ${preset.name}`); setName(""); dialog.current?.close()
    } catch { setError("Could not save the preset. Browser storage may be full.") }
  }
  const load = (preset: WorkspacePreset) => {
    try {
      onLoad(structuredClone(preset.value)); localStorage.setItem(LAST_PRESET_KEY, preset.id)
      setStatus(`Loaded ${preset.name}`); dialog.current?.close()
    } catch { setError("Could not load this preset.") }
  }
  return <section className="workspace-presets">
    <div className="workspace-preset-buttons">
      <button className="select" onClick={() => open("save")}>Save</button>
      <button className="select" onClick={() => open("load")}>Load</button>
    </div>
    {status && <span className="hint" role="status">{status}</span>}
    <dialog ref={dialog} className="workspace-preset-dialog" aria-labelledby="workspace-preset-title" onClick={event => {
      if (event.target === dialog.current) { const box = dialog.current!.getBoundingClientRect();
        if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.current?.close() }
    }}>
      <div className="workspace-preset-heading"><h2 id="workspace-preset-title">{mode === "save" ? "Save preset" : "Load preset"}</h2>
        <button className="icon-button" aria-label="Close presets" onClick={() => dialog.current?.close()}>×</button></div>
      {mode === "save" ? <form onSubmit={event => { event.preventDefault(); save() }}>
        <label>Preset name<input autoFocus className="number-input" value={name} onChange={event => setName(event.target.value)} /></label>
        <p className="hint">Saves the full setup, including policy, rendering, audio mappings, actuators, and sequences.</p>
        <button className="select" type="submit">Save preset</button>
      </form> : <div className="workspace-preset-list">
        {presets.length === 0 && <p className="hint">No presets yet. Save your current setup to create one.</p>}
        {presets.map(preset => <button key={preset.id} onClick={() => load(preset)}>
          <strong>{preset.name}</strong><span>{new Date(preset.savedAt).toLocaleString()} · {preset.value.sequences.length} sequences · {preset.value.audio.mappings.length} mappings</span>
        </button>)}
      </div>}
      {error && <p role="alert">{error}</p>}
    </dialog>
  </section>
}
