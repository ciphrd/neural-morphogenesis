"""Compact history keeps metrics, lazy replay, and incremental appends correct."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from history_index import compact_history, generation_record


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'history.jsonl'
        assert compact_history(path)['latestGeneration'] is None
        records=[{'generation':i,'best':1/(i+1),'mean':2.,'worst':3.,'allTimeBest':1/(i+1),
                  'seed':i,'optimizerState':{'sigma':.1/(i+1)},'weights':{'fc1w':[float(i)]*200},
                  'selectedSnapshotFitness':{'polar':{'losses':[[1]*100]}},'timing':{'seconds':i}}
                 for i in range(1100)]
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        payload=compact_history(path)
        assert len(payload['generations'])==1100
        assert len(payload['timings'])==1100
        assert all('weights' not in r and 'selectedSnapshotFitness' not in r for r in payload['generations'])
        assert payload['latestGeneration']==records[-1]
        import train_server
        with patch.object(train_server, 'HISTORY_PATH', path), \
             patch.object(train_server, '_run_dir_for_id', return_value=Path(tmp)):
            # Default endpoints must never silently truncate the chart to 500.
            assert len(train_server.history()['generations']) == 1100
            assert len(train_server.run_history('current')['generations']) == 1100
            assert len(train_server.run_history('archive')['generations']) == 1100
            assert train_server.history()['generations'][0]['generation'] == 0

        assert generation_record(path,0)==records[0]
        assert generation_record(path,1099)==records[-1]
        assert generation_record(path,9999) is None
        with patch('history_index.json.loads',side_effect=AssertionError('cached read reparsed history')):
            assert compact_history(path)==payload
        extra={**records[-1],'generation':1100}
        line=json.dumps(extra)+'\n'
        with path.open('a') as f:f.write(line[:12])
        assert len(compact_history(path)['generations'])==1100
        with path.open('a') as f:f.write(line[12:])
        assert compact_history(path)['latestGeneration']==extra
        assert generation_record(path,1100)==extra
        path.write_text(json.dumps(records[0])+'\n')
        assert len(compact_history(path)['generations'])==1
        replacement=Path(tmp)/'replacement'
        replacement.write_text(json.dumps(records[1])+'\n')
        replacement.replace(path)
        assert compact_history(path)['latestGeneration']==records[1]
        assert generation_record(path,0) is None
        import train_server
        with patch.object(train_server,'HISTORY_PATH',path), patch.object(train_server,'CHECKPOINTS_DIR',Path(tmp)), \
             patch.object(train_server,'_run_dir_for_id',return_value=Path(tmp)):
            assert train_server.history(compact=True)['latestGeneration']==records[1]
            assert train_server.run_history('archive',compact=True)['latestGeneration']==records[1]
            assert train_server.run_generation_record('current',1)==records[1]
            assert train_server.run_generation_record('archive',1)==records[1]
        print('[PASS] 1100-generation summaries, full sigma/timing history, lazy weights, cached/incomplete appends, file replacement and routes')


if __name__=='__main__': main()
