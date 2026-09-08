import { ProjectionView } from "./ProjectionView";
import { TrainingView } from "./TrainingView";

export function App() {
  if (new URLSearchParams(window.location.search).has("output")) {
    return <ProjectionView />;
  }
  return (
    <div className="app-shell">
      <div className="app-view">
        <TrainingView performanceMode />
      </div>
      <div id="performance-toolbar-host" />
    </div>
  );
}
