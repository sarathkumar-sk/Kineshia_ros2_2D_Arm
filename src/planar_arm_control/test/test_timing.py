import time

from planar_arm_control.timing import LoopTimer


def test_loop_timer_stats():
    lt = LoopTimer(0.01, window=100)
    for _ in range(30):
        lt.start()
        time.sleep(0.002)
        lt.stop()
        time.sleep(0.008)
    st = lt.stats()
    assert st["n"] == 29
    assert 8.0 < st["mean_ms"] < 20.0
    assert st["exec_mean_ms"] >= 1.5
    assert st["max_ms"] >= st["mean_ms"]
