"""All-generation timing backfill stays independent of weight-history limits."""
import json
import tempfile
from pathlib import Path
from train_server import _history_payload, MAX_HISTORY

def main():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'history.jsonl'
        assert _history_payload(path)=={'generations':[], 'timings':[]}
        with path.open('w') as stream:
            for generation in range(MAX_HISTORY+20):
                record={'generation':generation,'weights':[1,2,3]}
                if generation != 2:record['timing']={'seconds':generation}
                stream.write(json.dumps(record)+'\n')
            stream.write('{"generation":')
        payload=_history_payload(path)
        assert len(payload['generations'])==MAX_HISTORY
        assert len(payload['timings'])==MAX_HISTORY+19
        assert payload['timings'][0]['generation']==0
        assert all('weights' not in entry for entry in payload['timings'])
    print('[PASS] Full timing history, bounded weight history, missing timings and incomplete append')

if __name__=='__main__':main()
