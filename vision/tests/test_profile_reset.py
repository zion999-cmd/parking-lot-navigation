"""P0013.3: the learned appearance can be thrown away without restarting the process.

Measured on a long-running instance: profile_updates 9830, locked, and it had stopped following the car
while a fresh process found it at once. The profile locks after 25 updates and then keeps preferring the
look it learned, so a scene that changed - a different car, different light - is never re-learned.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import hero_belief                                               # noqa: E402


def taught():
    belief = hero_belief.HeroBelief(acquire_frames=1)
    belief.area = 120.0
    belief.width = 12.0
    belief.height = 9.0
    belief.aspect = 1.33
    belief.shape = 0.42
    belief.sat = 190.0
    belief.val = 175.0
    belief.hue = 4.0
    belief.profile_updates = 60
    belief.position = (100.0, 200.0)
    belief.state = hero_belief.TRACKING
    return belief


def test_reset_profile_forgets_the_appearance():
    belief = taught()
    belief.reset_profile()
    for name in ("area", "width", "height", "aspect", "shape", "sat", "val", "hue"):
        assert getattr(belief, name) is None, name
    assert belief.profile_updates == 0


def test_reset_profile_keeps_the_position_and_state():
    """Only the appearance is suspect. Where the car is and whether it is tracked are not."""
    belief = taught()
    belief.reset_profile()
    assert belief.position == (100.0, 200.0)
    assert belief.state == hero_belief.TRACKING


def test_a_locked_profile_reports_itself():
    belief = taught()
    assert belief.metrics()["hero_profile_locked"] is True
    belief.reset_profile()
    assert belief.metrics()["hero_profile_locked"] is False
    assert belief.metrics()["hero_profile_updates"] == 0
