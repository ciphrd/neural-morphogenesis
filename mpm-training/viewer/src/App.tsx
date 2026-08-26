import { useState } from "react";
import { LabView } from "./LabView";
import { TrainingView } from "./TrainingView";

export function App() {
  const [view, setView] = useState<"training" | "lab">("training");
  return (
    <div className="app-shell">
      <nav className="app-nav" aria-label="Workspace">
        <button className={view === "training" ? "is-active" : ""} onClick={() => setView("training")}>Training</button>
        <button className={view === "lab" ? "is-active" : ""} onClick={() => setView("lab")}>Lab</button>
      </nav>
      <div className="app-view">{view === "training" ? <TrainingView /> : <LabView />}</div>
    </div>
  );
}
