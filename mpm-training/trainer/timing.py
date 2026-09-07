"""Wall-clock timings; instrumentation never submits GPU work or synchronizes it."""
from contextlib import contextmanager
from time import perf_counter

class Timings:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.started = clock()
        self.stages = {}

    def add(self, name, seconds):
        stage = self.stages.setdefault(name, {'seconds': 0.0, 'count': 0, 'maxSeconds': 0.0})
        stage['seconds'] += seconds
        stage['count'] += 1
        stage['maxSeconds'] = max(stage['maxSeconds'], seconds)

    @contextmanager
    def measure(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.add(name, self.clock()-start)

    def report(self):
        return {'seconds': self.clock()-self.started,
                'stages': {name: dict(stage) for name, stage in self.stages.items()}}


def aggregate_rollouts(reports):
    reports = [report for report in reports if report]
    stages = {}
    for report in reports:
        for name, stage in report['stages'].items():
            total = stages.setdefault(name, {'seconds': 0.0, 'count': 0, 'maxSeconds': 0.0})
            total['seconds'] += stage['seconds']
            total['count'] += stage['count']
            total['maxSeconds'] = max(total['maxSeconds'], stage['maxSeconds'])
    durations = [report['seconds'] for report in reports]
    result = {'count': len(reports), 'seconds': sum(durations),
            'meanSeconds': sum(durations)/len(durations) if durations else 0.0,
            'maxSeconds': max(durations, default=0.0), 'stages': stages}

    gpu_reports = [report['gpu'] for report in reports if report.get('gpu', {}).get('samples', 0) > 0]
    if gpu_reports:
        gpu = aggregate_rollouts(gpu_reports)
        result['gpu'] = {'supported': True, 'samples': sum(r['samples'] for r in gpu_reports),
                         'seconds': gpu['seconds'], 'stages': gpu['stages'],
                         'droppedIntervals': sum(r.get('droppedIntervals', 0) for r in gpu_reports)}
    return result
