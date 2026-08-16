"""Provide privileged GT keypoints for kp_source=gt diagnostics."""

import numpy as np


def _quat_rotate_wxyz(quat, vec):
    w, x, y, z = [float(v) for v in quat]
    q = np.array([x, y, z])
    return vec + 2.0 * np.cross(q, np.cross(q, vec) + w * vec)


def _root(env, name):
    data = env.scene[name].data
    return (
        data.root_pos_w[0].cpu().numpy().astype(np.float64),
        data.root_quat_w[0].cpu().numpy().astype(np.float64),
    )


def _world_bbox(env, name):
    """Return the authored world bounds and prim position."""
    import re

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = env.scene[name].cfg.prim_path.replace(
        "{ENV_REGEX_NS}",
        "/World/envs/env_0",
    )
    path = re.sub(r"env_\.\*", "env_0", path)

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise SystemExit(f"[gt-points] no prim at {path} for {name}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    prim_pos = np.array(transform.ExtractTranslation())

    return np.array(box.GetMin()), np.array(box.GetMax()), prim_pos


_BODY_OFFSETS = {}


def _feature_from_body_offset(
    env,
    name,
    feature_world_authored,
    prim_pos_authored,
):
    """Re-anchor an authored feature offset at the live physics root."""
    key = (name, tuple(np.round(feature_world_authored, 4)))

    if key not in _BODY_OFFSETS:
        _BODY_OFFSETS[key] = (
            np.asarray(feature_world_authored)
            - np.asarray(prim_pos_authored)
        )

    pos, _ = _root(env, name)
    return pos + _BODY_OFFSETS[key]


def tea_gt_keypoints(env, grounded=None):
    """Return teapot-handle, spout, and teacup-rim keypoints."""
    pos, quat = _root(env, "teapot")
    handle = pos + _quat_rotate_wxyz(
        quat,
        np.array([0.0, 0.0566, -0.0624]),
    )
    mouth = pos + _quat_rotate_wxyz(
        quat,
        np.array([0.0, 0.0516, 0.0651]),
    )

    cup_lo, cup_hi, cup_prim = _world_bbox(env, "teacup")
    rim_authored = np.array(
        [
            (cup_lo[0] + cup_hi[0]) / 2.0,
            (cup_lo[1] + cup_hi[1]) / 2.0,
            cup_hi[2],
        ]
    )
    cup_rim = _feature_from_body_offset(
        env,
        "teacup",
        rim_authored,
        cup_prim,
    )

    keypoints = np.stack([handle, mouth, cup_rim]).astype(np.float32)
    metadata = {
        "owners": ["teapot", "teapot", "teacup"],
        "virtual": set(),
        "grasp_extent": {0: 0.010},
    }

    print(
        f"[gt-points] tea kps "
        f"handle={np.round(handle, 3)} "
        f"mouth={np.round(mouth, 3)} "
        f"cup={np.round(cup_rim, 3)} "
        f"teapot_root={np.round(pos, 3)}",
        flush=True,
    )
    return keypoints, metadata


def pot_gt_keypoints(env, grounded=None):
    """Return lid-rim, egg, and pot-rim keypoints."""
    cover_lo, cover_hi, cover_prim = _world_bbox(env, "cover")
    cover_center = (cover_lo + cover_hi) / 2.0
    rim_authored = np.array(
        [
            cover_hi[0],
            cover_center[1],
            cover_center[2],
        ]
    )
    rim = _feature_from_body_offset(
        env,
        "cover",
        rim_authored,
        cover_prim,
    )

    egg_pos, _ = _root(env, "egg")

    pot_lo, pot_hi, pot_prim = _world_bbox(env, "pot")
    pot_rim_authored = np.array(
        [
            (pot_lo[0] + pot_hi[0]) / 2.0,
            (pot_lo[1] + pot_hi[1]) / 2.0,
            pot_hi[2],
        ]
    )
    pot_rim = _feature_from_body_offset(
        env,
        "pot",
        pot_rim_authored,
        pot_prim,
    )

    keypoints = np.stack([rim, egg_pos, pot_rim]).astype(np.float32)
    metadata = {
        "owners": ["cover", "egg", "pot"],
        "virtual": set(),
        "grasp_extent": {0: 0.015},
    }

    print(
        f"[gt-points] pot kps "
        f"lid_rim={np.round(rim, 3)} "
        f"egg={np.round(egg_pos, 3)} "
        f"pot_rim={np.round(pot_rim, 3)}",
        flush=True,
    )
    return keypoints, metadata


def weight_gt_keypoints(env, grounded=None):
    """Return pear, apple, and scale-top keypoints."""
    pear, _ = _root(env, "pear")
    apple, _ = _root(env, "apple")

    scale_lo, scale_hi, scale_prim = _world_bbox(env, "scale")
    top_authored = np.array(
        [
            (scale_lo[0] + scale_hi[0]) / 2.0,
            (scale_lo[1] + scale_hi[1]) / 2.0,
            scale_hi[2],
        ]
    )
    scale_top = _feature_from_body_offset(
        env,
        "scale",
        top_authored,
        scale_prim,
    )

    keypoints = np.stack([pear, apple, scale_top]).astype(np.float32)
    metadata = {
        "owners": ["pear", "apple", "scale"],
        "virtual": set(),
    }

    print(
        f"[gt-points] weight kps "
        f"pear={np.round(pear, 3)} "
        f"apple={np.round(apple, 3)} "
        f"scale_top={np.round(scale_top, 3)}",
        flush=True,
    )
    return keypoints, metadata


GT_KEYPOINTS = {
    "tea": tea_gt_keypoints,
    "pot": pot_gt_keypoints,
    "weight": weight_gt_keypoints,
}