import { TrainingView } from "./TrainingView";

// Single view — no interactive/training mode switch (unlike
// trainer/frontend's App.tsx): this project only ever ships the passive
// training viewer, per its own scope (see README.md's "Visualization"
// section).
export default function App() {
  return (
    <div className="app">
      <TrainingView />
    </div>
  );
}
