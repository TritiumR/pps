"""Seat the lid on the pot for the in-repo Isaac-Pot-* tasks.

The pot (``E_pot1_1``) and cover (``E_cover_2``) in the pot task wrap nested kitchen prims
(``model_kitchenware006`` inside ``Interactive_kitchen_with_parlor.usd``). Their PhysX
bodies spawn ~0.52m off their room-shifted USD geometry, so the *kinematic* pot renders at
its USD mesh while the *dynamic* cover is driven to its PhysX pose -- leaving the lid
visibly detached, floating to the side of the pot.

A pose write issued during the env *reset event* is silently dropped for these nested
bodies (verified), but a write issued *mid-episode* (after >=1 physics step) sticks. So the
fix can't live in the (shared) task config; the front-ends call this helper right after
their post-reset settle loop instead. It writes the kinematic pot to its declared
``init_state`` pose -- moving its collider to match its visual -- drops the dynamic cover
just above the pot, and settles so the lid rests on it.

No-ops on any task whose scene lacks both a ``pot`` and a ``cover`` rigid object, so it is
safe to call unconditionally after every reset.
"""

import torch


def seat_pot_lid(env, hold_action, n_settle=25, drop_height=0.13):
    """Re-seat the pot lid. Returns True if it acted (pot task), False otherwise.

    Args:
        env: the unwrapped manager-based env (already reset + settled at least one step).
        hold_action: a do-nothing action for this task's controller; used to step while the
            dropped cover settles. Shape (action_dim,) or (num_envs, action_dim).
        n_settle: physics steps to let the cover settle onto the pot.
        drop_height: metres above the pot origin to release the cover from.
    """
    rigid = getattr(env.scene, "rigid_objects", {}) or {}
    if "pot" not in rigid or "cover" not in rigid:
        return False

    pot = rigid["pot"]
    cover = rigid["cover"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    origins = env.scene.env_origins  # default_root_state pose is in the env-local frame

    pot_state = pot.data.default_root_state.clone()
    cover_state = cover.data.default_root_state.clone()
    pot_state[:, :3] += origins
    cover_state[:, :3] += origins
    # Release the cover from just above the pot rim so it drops cleanly onto the pot
    # rather than spawning inside the pot collider and being ejected sideways.
    cover_state[:, 2] = pot_state[:, 2] + drop_height

    pot.write_root_pose_to_sim(pot_state[:, :7], env_ids=env_ids)
    pot.write_root_velocity_to_sim(pot_state[:, 7:], env_ids=env_ids)
    cover.write_root_pose_to_sim(cover_state[:, :7], env_ids=env_ids)
    cover.write_root_velocity_to_sim(cover_state[:, 7:], env_ids=env_ids)

    hold = torch.as_tensor(hold_action, dtype=torch.float32, device=env.device)
    if hold.ndim == 1:
        hold = hold.unsqueeze(0)
    for _ in range(n_settle):
        env.step(hold)
    return True
