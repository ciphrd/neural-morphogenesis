"""Incremental compact history index; replay weights are read on demand."""
from collections import OrderedDict
import json
from pathlib import Path
from threading import RLock

_KEYS = ('generation', 'best', 'mean', 'worst', 'allTimeBest', 'seed', 'optimizerState')
_LOCK = RLock()
_CACHES = OrderedDict()


class HistoryIndex:
    def __init__(self):
        self.position = 0
        self.identity = None
        self.stamp = None
        self.summaries = {}
        self.timings = {}
        self.offsets = {}
        self.latest = None

    def refresh(self, path):
        if not path.is_file():
            self.__init__()
            return
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        stamp = (stat.st_size, stat.st_mtime_ns)
        if identity != self.identity or stat.st_size < self.position or (self.stamp and stat.st_size == self.stamp[0] and stamp != self.stamp):
            self.__init__()
        if stamp == self.stamp:
            return
        self.identity, self.stamp = identity, stamp
        with path.open('rb') as stream:
            stream.seek(self.position)
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break  # Retry an incomplete concurrent append next time.
                self.position = stream.tell()
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                generation = record.get('generation')
                if not isinstance(generation, int):
                    continue
                self.summaries[generation] = {key: record[key] for key in _KEYS if key in record}
                self.offsets[generation] = offset
                if record.get('timing') is not None:
                    self.timings[generation] = {'generation': generation, 'timing': record['timing']}
                if self.latest is None or generation >= self.latest['generation']:
                    self.latest = record


def _index(path):
    key = str(Path(path).resolve())
    index = _CACHES.pop(key, None) or HistoryIndex()
    _CACHES[key] = index
    while len(_CACHES) > 4:
        _CACHES.popitem(last=False)
    index.refresh(Path(path))
    return index


def compact_history(path):
    with _LOCK:
        index = _index(path)
        return {'generations': [index.summaries[k] for k in sorted(index.summaries)],
                'timings': [index.timings[k] for k in sorted(index.timings)],
                'latestGeneration': index.latest}


def generation_record(path, generation):
    with _LOCK:
        index = _index(path)
        offset = index.offsets.get(generation)
        if offset is None:
            return None
        with Path(path).open('rb') as stream:
            stream.seek(offset)
            return json.loads(stream.readline())
