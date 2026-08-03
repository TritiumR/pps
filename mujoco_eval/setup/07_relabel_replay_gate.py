"""G3: closed-loop replay of the relabeled absolute joint targets against the demo.

Resets held-out demos to their start state and drives apply_arm with
joint_actions[t] := robot0_joint_pos[t+1] plus the demo gripper command. The sim is never
resynced, so divergence either compounds or it does not.

Gate, pre-declared: pooled TCP p95 < 15 mm and task success >= 4/5.

    python setup/07_relabel_replay_gate.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import h5py
import numpy as np

_MG = pathlib.Path(__file__).resolve().parent.parent
if str(_MG) not in sys.path:
    sys.path.insert(0, str(_MG))

from ..env.mujoco_env import MuJoCoEnv  # noqa: E402

DATA = _MG / "data/stack_d0/demo.hdf5"
FK_FIT = _MG / "bench/fk_fits/fk_fit_stack_d0.json"
OUT = _MG / "bench/fk_fits/g3_relabel_replay_stack.json"
OBJ = {"cubeA": (0, 3), "cubeB": (7, 10)}
TCP_P95_MM = 15.0
MIN_SUCCESS = 4


def replay(mu, demo):
    q = np.asarray(demo["obs/robot0_joint_pos"], dtype=np.float64)
    act = np.asarray(demo["actions"], dtype=np.float64)
    eef = np.asarray(demo["obs/robot0_eef_pos"], dtype=np.float64)
    obj = np.asarray(demo["obs/object"], dtype=np.float64)
    states = np.asarray(demo["states"])
    T = act.shape[0] - 1

    mu.env.reset_to({"states": states[0], "model": demo.attrs["model_file"]})
    mu._bind()

    joint_err, tcp_mm, obj_mm, success = [], [], {k: [] for k in OBJ}, False
    for t in range(T):
        mu.apply_arm(q[t + 1], grip_close=act[t, 6] > 0)
        joint_err.append(float(np.abs(mu.q0().numpy() - q[t + 1]).max()))
        tcp_mm.append(float(np.linalg.norm(mu.tcp() - eef[t + 1]) * 1e3))
        for name, (a, b) in OBJ.items():
            sim_pos, _ = mu.object_pose(name)
            obj_mm[name].append(float(np.linalg.norm(sim_pos - obj[t + 1, a:b]) * 1e3))
        success = success or mu.success()
    return {
        "T": T,
        "success": bool(success),
        "joint_err_rad": {"p95": float(np.percentile(joint_err, 95)),
                          "max": float(np.max(joint_err))},
        "tcp_mm": {"p50": float(np.percentile(tcp_mm, 50)),
                   "p95": float(np.percentile(tcp_mm, 95)),
                   "max": float(np.max(tcp_mm))},
        "obj_final_mm": {k: v[-1] for k, v in obj_mm.items()},
        "obj_max_mm": {k: float(np.max(v)) for k, v in obj_mm.items()},
        "_tcp_series": tcp_mm,
    }


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mu = MuJoCoEnv(str(DATA), str(FK_FIT))
    mu.env.reset()          # first bind
    mu._bind()
    results, pooled = {}, []
    with h5py.File(DATA, "r") as f:
        for i in range(first, first + n):
            name = f"demo_{i}"
            r = replay(mu, f[f"data/{name}"])
            pooled.extend(r.pop("_tcp_series"))
            results[name] = r
            print(f"{name}: T={r['T']} success={r['success']} "
                  f"joint max={r['joint_err_rad']['max']:.4f} rad  "
                  f"tcp p95={r['tcp_mm']['p95']:.1f} max={r['tcp_mm']['max']:.1f} mm  "
                  f"objA final={r['obj_final_mm']['cubeA']:.1f} "
                  f"objB final={r['obj_final_mm']['cubeB']:.1f} mm", flush=True)
    n_succ = sum(r["success"] for r in results.values())
    pooled_p95 = float(np.percentile(pooled, 95))
    verdict = "PASS" if (pooled_p95 < TCP_P95_MM and n_succ >= MIN_SUCCESS) else "FAIL"
    print(f"\nG3 GATE [stack relabel-replay, demos {first}-{first + n - 1}]: "
          f"pooled TCP p95 = {pooled_p95:.1f} mm (rule < {TCP_P95_MM}), "
          f"success {n_succ}/{n} (rule >= {MIN_SUCCESS}) -> {verdict}")
    payload = {"gate": {"pooled_tcp_p95_mm": pooled_p95, "tcp_rule_mm": TCP_P95_MM,
                        "success": n_succ, "n": n, "success_rule": MIN_SUCCESS,
                        "verdict": verdict},
               "demos": results}
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
