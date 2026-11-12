import numpy as np

from planar_arm_control.pick_place import MotionInterface, Payload, PickPlaceSequencer


class FakeMotion(MotionInterface):
    """Instant-ish motions; records the call sequence."""

    def __init__(self, steps_per_move=3, reject=None):
        self.log, self.ee = [], np.zeros(2)
        self.payload = Payload((4.0, 2.0))
        self._remaining = 0
        self._goal = None
        self.steps = steps_per_move
        self.reject = reject or set()

    def check_target(self, xy, q_from):
        ok = tuple(xy) not in self.reject
        return ok, np.zeros(3), "ok" if ok else "unreachable"

    def start_move(self, xy):
        self.log.append(("move", tuple(xy)))
        self._goal, self._remaining = np.asarray(xy, float), self.steps
        return True, "ok"

    def start_home(self):
        self.log.append(("home",))
        self._goal, self._remaining = np.array([0.0, 6.5]), self.steps
        return True, "ok"

    def motion_done(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self.ee = self._goal
            self.payload.follow(self.ee)
            return True
        return False

    def motion_progress(self):
        return 0.5

    def stop(self):
        self.log.append(("stop",))

    def set_gripper(self, closed):
        self.log.append(("grip", closed))
        if closed:
            return self.payload.try_attach(self.ee)
        self.payload.release(self.ee)
        return False

    def current_q(self):
        return np.zeros(3)

    def end_effector(self):
        return tuple(self.ee)


def seq(m):
    return PickPlaceSequencer(m, grip_time=0.01, poll_period=0.0, feedback_period=0.0)


def test_full_sequence_order_and_payload():
    m, fbs = FakeMotion(), []
    out = seq(m).run((4.0, 2.0), (-3.0, 3.0), return_home=True,
                     publish_feedback=fbs.append)
    assert out.success, out.message
    assert m.log == [("move", (4.0, 2.0)), ("grip", True),
                     ("move", (-3.0, 3.0)), ("grip", False), ("home",)]
    np.testing.assert_allclose(m.payload.position, (-3.0, 3.0))
    assert fbs[-1].total_progress == 1.0
    phases = [f.phase for f in fbs]
    assert phases.index("move_to_pick") < phases.index("grasp") < \
        phases.index("move_to_place") < phases.index("release")


def test_unreachable_pick_rejected_before_moving():
    m = FakeMotion(reject={(7.0, 3.0)})
    out = seq(m).run((7.0, 3.0), (-3.0, 3.0))
    assert not out.success and "pick target rejected" in out.message
    assert m.log == []


def test_cancel_stops_motion():
    m = FakeMotion(steps_per_move=1000)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 5
    out = seq(m).run((4.0, 2.0), (-3.0, 3.0), is_cancel_requested=cancel)
    assert out.canceled and not out.success
    assert ("stop",) in m.log


def test_grasp_fails_if_object_not_there():
    m = FakeMotion()
    m.payload = Payload((0.0, 6.0))
    out = seq(m).run((4.0, 2.0), (-3.0, 3.0))
    assert not out.success and "no object" in out.message
