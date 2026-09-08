import type { GenerationTiming, RolloutTiming, TimingEntry, TimingStage } from "../gpu/types";

function averageReports(reports: RolloutTiming[]): RolloutTiming {
  const stages: Record<string, TimingStage> = {};
  for (const report of reports) for (const [name, stage] of Object.entries(report.stages)) {
    const sum = stages[name] ??= { seconds: 0, count: 0, maxSeconds: 0 };
    sum.seconds += stage.seconds;
    sum.count += stage.count;
    sum.maxSeconds = Math.max(sum.maxSeconds, stage.maxSeconds);
  }
  for (const sum of Object.values(stages)) { sum.seconds /= reports.length; sum.count /= reports.length; }
  const gpuReports = reports.flatMap(r => r.gpu && r.gpu.samples > 0 ? [r.gpu] : []);
  let gpu;
  if (gpuReports.length) {
    const gpuStages: Record<string, TimingStage> = {};
    for (const r of gpuReports) for (const [key, stage] of Object.entries(r.stages)) {
      const sum = gpuStages[key] ??= {seconds: 0, count: 0, maxSeconds: 0};
      sum.seconds += stage.seconds; sum.count += stage.count;
      sum.maxSeconds = Math.max(sum.maxSeconds, stage.maxSeconds);
    }
    gpu = {supported: true, samples: gpuReports.reduce((s,r) => s+r.samples,0),
      seconds: gpuReports.reduce((s,r) => s+r.seconds,0), stages: gpuStages,
      droppedIntervals: gpuReports.reduce((s,r) => s+(r.droppedIntervals ?? 0),0)};
  }
  return { seconds: reports.reduce((s, r) => s + r.seconds, 0) / reports.length, stages, gpu };
}

export function averageTimings(history: TimingEntry[]): GenerationTiming | undefined {
  if (!history.length) return undefined;
  const values = history.map(h => h.timing);
  const mean = (get: (t: GenerationTiming) => number) => values.reduce((sum, t) => sum + get(t), 0) / values.length;
  const rollouts = averageReports(values.map(t => t.rollouts));
  const winners = values.flatMap(t => t.winner ? [t.winner] : []);
  const count = mean(t => t.rollouts.count);
  return {
    seconds: mean(t => t.seconds), poolSeconds: mean(t => t.poolSeconds),
    selectionSeconds: mean(t => t.selectionSeconds), previewSeconds: mean(t => t.previewSeconds),
    checkpointSeconds: mean(t => t.checkpointSeconds), otherSeconds: mean(t => t.otherSeconds),
    rollouts: { ...rollouts, count, meanSeconds: count ? rollouts.seconds / count : 0,
      maxSeconds: Math.max(...values.map(t => t.rollouts.maxSeconds)) },
    winner: winners.length ? averageReports(winners) : undefined,
  };
}

export function timingReport(timing: GenerationTiming, scope: string): RolloutTiming | undefined {
  const report = scope.endsWith("winner") ? timing.winner : timing.rollouts;
  if (!scope.startsWith("gpu")) return report;
  const gpu = report?.gpu;
  if (!gpu?.samples) return undefined;
  return {seconds: gpu.seconds/gpu.samples, gpu,
    stages: Object.fromEntries(Object.entries(gpu.stages).map(([key, stage]) => [key,
      {...stage, seconds:stage.seconds/gpu.samples, count:stage.count/gpu.samples}]))};
}
