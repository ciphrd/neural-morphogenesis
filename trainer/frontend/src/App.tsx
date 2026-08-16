import { useState } from "react";
import { InteractiveView } from "./InteractiveView";
import { TrainingView } from "./TrainingView";

type Mode = "interactive" | "training";

export default function App() {
  const [mode, setMode] = useState<Mode>("interactive");

  return (
    <div className="app">
      <div className="mode-switch">
        <button className={mode === "interactive" ? "active" : ""} onClick={() => setMode("interactive")}>
          Interactive
        </button>
        <button className={mode === "training" ? "active" : ""} onClick={() => setMode("training")}>
          Training
        </button>
      </div>
      {mode === "interactive" ? <InteractiveView /> : <TrainingView />}
    </div>
  );
}
