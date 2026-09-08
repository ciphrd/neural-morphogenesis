"""Check physics chunk barriers without constructing a GPU device."""
from unittest.mock import MagicMock
from mpm_core import MpmCore

def main():
    for substeps, wait, expected in ((0,False,0),(16,True,1),(16,False,0),
                                    (128,False,0),(129,False,1),(256,False,1),
                                    (300,False,2),(300,True,3)):
        core=MagicMock()
        core._active_count=40
        core._repulsion_enabled=False
        core._MAX_SUBSTEPS_PER_SUBMIT=128
        MpmCore.step(core,substeps,wait_for_completion=wait)
        assert core.device.queue.read_buffer.call_count==expected,(substeps,wait)
        assert core.device.queue.submit.call_count==(substeps+127)//128
    print('[PASS] Default synchronous completion; deferred final wait preserves all intermediate barriers')

if __name__=='__main__':main()
