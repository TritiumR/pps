"""Faithful port of ReKep's grounding front-end (keypoint proposal + VLM constraint
generation) plus the IsaacLab glue to run it on the PPS tasks.

See REKEP.md / the project plan. The keypoint proposer and constraint generator are
ported as closely as possible from the upstream ReKep repo; the obs adapter,
visualization, and solvers are rebuilt against IsaacLab/Franka.
"""
