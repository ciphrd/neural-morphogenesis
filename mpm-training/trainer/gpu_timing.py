"""Sampled native GPU intervals. Timestamp readback cost stays in host timings."""
from contextlib import contextmanager
import math
import numpy as np
import wgpu
from timing import Timings

class GpuTimings:
    QUERY_COUNT = 2048

    def __init__(self, device, interval=20):
        self.device = device
        self.interval = interval
        self.supported = all(name in device.features for name in ('timestamp-query', 'timestamp-query-inside-encoders'))
        self.query_set = self.buffer = None
        if self.supported:
            # wgpu-native exposes raw ticks unless automatic normalization is on.
            # Respect its actual period instead of assuming Metal's usual 1 ns.
            from wgpu.backends.wgpu_native._ffi import lib
            self.period_ns = float(lib.wgpuQueueGetTimestampPeriod(device.queue._internal))
            self.supported = math.isfinite(self.period_ns) and self.period_ns > 0
        if self.supported:
            self.buffer = device.create_buffer(size=self.QUERY_COUNT*8,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
        self.reset()

    def reset(self):
        self.steps = self.samples = self.dropped = 0
        self.active = False
        self.pending = []
        self.totals = Timings()

    def begin_sample(self):
        self.steps += 1
        self.active = self.supported and self.interval > 0 and (self.steps-1) % self.interval == 0
        self.pending = []
        if self.active:
            # Fresh slots avoid stale timestamp availability across submissions on Metal.
            if self.query_set is not None:
                self.query_set.destroy()
            self.query_set = self.device.create_query_set(type="timestamp", count=self.QUERY_COUNT)

    @contextmanager
    def measure(self, encoder, name):
        if not self.active:
            yield
            return
        index = len(self.pending)*2
        if index+2 > self.QUERY_COUNT:
            self.dropped += 1
            yield
            return
        from wgpu.backends.wgpu_native.extras import write_timestamp
        self.pending.append(name)
        write_timestamp(encoder, self.query_set, index)
        try:
            yield
        finally:
            write_timestamp(encoder, self.query_set, index+1)

    def begin_compute_pass(self, encoder, name):
        """Use pass-boundary timestamps for fine intervals (ordered by the backend)."""
        if not self.active:
            return encoder.begin_compute_pass()
        index = len(self.pending)*2
        if index+2 > self.QUERY_COUNT:
            self.dropped += 1
            return encoder.begin_compute_pass()
        self.pending.append(name)
        return encoder.begin_compute_pass(timestamp_writes={
            "query_set": self.query_set,
            "beginning_of_pass_write_index": index,
            "end_of_pass_write_index": index+1,
        })

    def begin_render_pass(self, encoder, name, **descriptor):
        if self.active:
            index = len(self.pending)*2
            if index+2 <= self.QUERY_COUNT:
                self.pending.append(name)
                descriptor["timestamp_writes"] = {"query_set": self.query_set,
                    "beginning_of_pass_write_index": index, "end_of_pass_write_index": index+1}
            else:
                self.dropped += 1
        return encoder.begin_render_pass(**descriptor)

    def finish_sample(self):
        if not self.active or not self.pending:
            return
        count = len(self.pending)*2
        encoder = self.device.create_command_encoder()
        encoder.resolve_query_set(self.query_set, 0, count, self.buffer, 0)
        self.device.queue.submit([encoder.finish()])
        ticks = np.frombuffer(self.device.queue.read_buffer(self.buffer, 0, count*8), dtype=np.uint64)
        for name, start, end in zip(self.pending, ticks[::2], ticks[1::2]):
            if end >= start:
                self.totals.add(name, (int(end)-int(start))*self.period_ns*1e-9)
            else:
                self.dropped += 1
        self.samples += 1
        self.active = False

    def report(self):
        stages = self.totals.report()['stages']
        return {'supported': self.supported, 'interval': self.interval, 'samples': self.samples,
                'droppedIntervals': self.dropped, 'stages': stages,
                'seconds': sum(stage['seconds'] for stage in stages.values())}
