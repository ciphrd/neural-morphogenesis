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
        <div className="app-nav-tabs">
          <button className={view === "training" ? "is-active" : ""} onClick={() => setView("training")}>Training</button>
          <button className={view === "lab" ? "is-active" : ""} onClick={() => setView("lab")}>Lab</button>
          <button className={view === "performance" ? "is-active" : ""} onClick={() => setView("performance")}>Performance</button>
        </div>
        <div id="training-header-actions" className="app-nav-actions" />
      </nav>
      <div className="app-view">
        {view === "lab" ? <LabView /> : <TrainingView performanceMode={view === "performance"} />}
      </div>
    </div>
  );
}
