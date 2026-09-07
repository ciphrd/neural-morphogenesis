"""Deterministic checks for timing accumulation and parallel aggregation."""
import json
from timing import Timings, aggregate_rollouts

def main():
    now=[10.0]
    timings=Timings(clock=lambda:now[0])
    with timings.measure('fitness'):
        now[0]+=2
    with timings.measure('fitness'):
        now[0]+=3
    try:
        with timings.measure('readback'):
            now[0]+=1
            raise ValueError('simulated failure')
    except ValueError:
        pass
    report=timings.report()
    assert report['seconds']==6
    assert report['stages']['fitness']==dict(seconds=5.,count=2,maxSeconds=3.)
    combined=aggregate_rollouts([report,report,None])
    assert combined['count']==2 and combined['seconds']==12 and combined['meanSeconds']==6
    assert combined['stages']['fitness']==dict(seconds=10.,count=4,maxSeconds=3.)
    assert combined['maxSeconds']==6
    assert aggregate_rollouts([])['count']==0
    json.dumps(combined,allow_nan=False)
    print('[PASS] Timing accumulation, exceptions, sums/counts/maxima, empty aggregation and JSON serialization')

if __name__=='__main__':main()
