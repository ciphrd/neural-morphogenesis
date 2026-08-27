import { useState } from "react";
import { LabView } from "./LabView";
import { ProjectionView } from "./ProjectionView";
import { TrainingView } from "./TrainingView";

export function App() {
  const [view, setView] = useState<"training" | "lab" | "performance">("training");
  if (new URLSearchParams(window.location.search).has("output")) {
    return <ProjectionView />;
  }
  return (
    <div className="app-shell">
      <nav className="app-nav" aria-label="Workspace">
        <button className={view === "training" ? "is-active" : ""} onClick={() => setView("training")}>Training</button>
        <button className={view === "lab" ? "is-active" : ""} onClick={() => setView("lab")}>Lab</button>
        <button className={view === "performance" ? "is-active" : ""} onClick={() => setView("performance")}>Performance</button>
      </nav>
      <div className="app-view">
        {view === "lab" ? <LabView /> : <TrainingView performanceMode={view === "performance"} />}
      </div>
    </div>
  );
}
