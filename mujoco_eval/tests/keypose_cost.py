"""The contact cost must respond to what a point-to-point cost is blind to.

That is its entire justification. `rekep_keypose` measures ||TCP - keypoint||, which is unchanged
by rotating the gripper about the approach axis or by where the fingers sit -- and those are the
degrees of freedom a keypose proposal perturbs. Measured: the keypose term alone separated a
128-proposal cloud by 5.8% against the full composite's 19.7%, so "fewer terms" is not the fix and
a richer geometric signal has to be.
"""

from __future__ import annotations

import torch

from mujoco_eval.steering.keypose_cost import (finger_samples, keypose_contact_cost, track_cost)


class _Geom:
    open_half = 0.04
    tcp_to_tip = 0.0
    finger_r = 0.012


def _quat(yaw):
    """wxyz quaternion for a rotation about z."""
    return torch.tensor([[float(torch.cos(torch.tensor(yaw / 2))), 0.0, 0.0,
                          float(torch.sin(torch.tensor(yaw / 2)))]])


def test_samples_straddle_the_tcp():
    """The two finger faces sit either side of the TCP, one open_half out on each."""
    pos = torch.zeros(1, 3)
    s = finger_samples(pos, _quat(0.0), _Geom(), per_finger=3)
    assert s.shape == (1, 6, 3)
    # Half the samples on one side of the closing axis, half on the other.
    y = s[0, :, 1]
    assert torch.allclose(y.abs(), torch.full_like(y, _Geom.open_half), atol=1e-5)
    assert (y > 0).sum() == 3 and (y < 0).sum() == 3


def test_surface_not_centre():
    """Subtracting the radius makes the cost vanish AT the surface, not at the centre.

    per_finger=1 puts the single sample on each face at the knuckle, exactly open_half out; with
    more samples they spread along the finger and cannot all lie on a sphere at once, so the
    floor is small but nonzero -- which is the geometry, not a defect.
    """
    pos = torch.zeros(1, 3)
    radius = 0.04                       # exactly the finger half-opening
    at_surface = keypose_contact_cost(pos, _quat(0.0), (0.0, 0.0, 0.0), radius, _Geom(),
                                      per_finger=1)
    assert float(at_surface) < 1e-5, float(at_surface)
    # A centre metric would score this identically; a surface metric must not.
    away = keypose_contact_cost(pos + torch.tensor([[0.0, 0.0, 0.05]]), _quat(0.0),
                                (0.0, 0.0, 0.0), radius, _Geom())
    assert float(away) > float(at_surface) + 0.01


def test_responds_to_yaw():
    """A point-to-point cost is flat in yaw; this one is not, for an off-axis target."""
    pos = torch.zeros(1, 3)
    target = (0.03, 0.0, 0.0)
    a = float(keypose_contact_cost(pos, _quat(0.0), target, 0.02, _Geom()))
    b = float(keypose_contact_cost(pos, _quat(1.5707963), target, 0.02, _Geom()))
    assert abs(a - b) > 1e-3, (a, b)


def test_track_cost_zero_when_rows_hold_the_keypose():
    """Rows already at the keypose incur no tracking penalty; rows away from it do."""
    chunk = torch.zeros(2, 5, 8)
    assert torch.allclose(track_cost(chunk, 4), torch.zeros(2))
    chunk[1, :4, :7] = 1.0
    assert float(track_cost(chunk, 4)[1]) > 0.9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all keypose contact-cost tests passed")
