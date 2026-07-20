"""Train the geometry-conditioned score proxies for score-space PPS (Config B).

  reference: distill the base's on-policy score along its own denoise trajectory (paper Eq. 5),
             from the ref_s*.npz labels the base runner records with --record_ref.
  task:      init from the reference, then score-match demo actions (paper Eq. 6), conditioned on the
             ctx recovered from each demo by grounding-replay.

Run:
  python -m vlm_base.train_geom_proxy --mode reference \
    --labels "data/weight_ref_labels/ref_s*.npz" --out openpi/checkpoints/geom_proxy_weight/reference
  python -m vlm_base.train_geom_proxy --mode task --task weight \
    --init openpi/checkpoints/geom_proxy_weight/reference/model.pt \
    --demos data/weight/generated_dataset.hdf5 --out openpi/checkpoints/geom_proxy_weight/task
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vlm_base.geom_proxy import GeomProxyConfig, GeomScoreProxy, features_from_geom, time_cond_for


class GeomRefDataset(Dataset):
    """One item per recorded denoise-path label: the (x_t, time) state paired with its chunk's ctx geometry."""

    def __init__(self, files, num_train_timesteps=100):
        self.items = []
        scores = []
        for f in sorted(files):
            d = np.load(f)
            pos_of = {int(c): i for i, c in enumerate(d["obs_chunk"])}   # chunk number -> obs row
            for i in range(d["x"].shape[0]):
                p = pos_of[int(d["chunk"][i])]
                feat = features_from_geom(
                    d["obs_obj_pos"][p], d["obs_obj_ext"][p], d["obs_grasp_idx"][p],
                    d["obs_target"][p], d["obs_eef"][p], d["obs_stage"][p], d["obs_joint"][p])
                feat["x"] = d["x"][i].astype(np.float32)
                feat["score"] = d["score"][i].astype(np.float32)
                feat["time"] = np.float32(
                    time_cond_for(int(d["it"][i]), int(d["num_iters"][i]), num_train_timesteps))
                self.items.append(feat)
                scores.append(d["score"][i])
        self.score_std = float(np.std(np.concatenate([s.reshape(-1) for s in scores]))) or 1.0
        self.num_objects = self.items[0]["obj_feats"].shape[0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {k: torch.as_tensor(v) for k, v in self.items[i].items()}


class GeomTaskDataset(Dataset):
    """One item per demo action window: the model-space demo chunk paired with its window's ctx geometry.

    The ctx (phase, grasp target, reach target) is recovered from the demo by grounding-replay and the demo
    actions are inverted into the base's normalized delta space (see vlm_base.demo_ctx).
    """

    def __init__(self, demo_path, obj_names, extents, a_q01, a_q99, horizon, grasp_objs, place_obj):
        import h5py

        from vlm_base.demo_ctx import demo_task_items
        self.items = []
        with h5py.File(demo_path, "r") as f:
            root = f["data"] if "data" in f else f
            for key in root.keys():
                for feat, clean in demo_task_items(root[key], obj_names, extents, a_q01, a_q99,
                                                   horizon, grasp_objs, place_obj):
                    feat["actions"] = clean
                    self.items.append(feat)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {k: torch.as_tensor(v) for k, v in self.items[i].items()}


def _wandb_init(args, cfg, role):
    """Start a wandb run when --wandb is set, else return None. Uses the login's default entity if unset."""
    if not args.wandb:
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name or f"{role}-{args.task}",
        config={"role": role, "task": args.task, "lr": args.lr, "batch_size": args.batch_size,
                "epochs": args.epochs, **vars(cfg)})


def _run_training(model, ds, args, mode, device, run=None):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        running = n = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.loss(batch, mode=mode).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = batch["obj_feats"].shape[0]
            running += float(loss) * bs
            n += bs
        epoch_mse = running / max(n, 1)
        if run is not None:
            run.log({"epoch": epoch, "mse": epoch_mse})
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"[geom_proxy] epoch {epoch:4d}  mse={epoch_mse:.4g}", flush=True)


def _save(model, cfg, out, role):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "model.pt")
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, path)
    print(f"[geom_proxy] saved {role} proxy -> {path}", flush=True)


def train_reference(args):
    files = sorted(glob.glob(args.labels))
    if not files:
        raise SystemExit(f"[geom_proxy] no label files match {args.labels!r}")
    ds = GeomRefDataset(files)
    print(f"[geom_proxy] {len(ds)} labels from {len(files)} files; score_std={ds.score_std:.1f} "
          f"num_objects={ds.num_objects}", flush=True)
    cfg = GeomProxyConfig(score_scale=ds.score_std, predict=args.predict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeomScoreProxy(cfg).to(device)
    run = _wandb_init(args, cfg, "reference")
    _run_training(model, ds, args, "reference", device, run)
    _save(model, cfg, args.out, "reference")
    if run is not None:
        run.finish()


def train_task(args):
    if not args.demos:
        raise SystemExit("[geom_proxy] task mode needs --demos <hdf5>")
    ref = np.load(sorted(glob.glob(args.labels))[0])   # object names + static extents, as the reference saw them
    obj_names = [str(n) for n in ref["obj_names"]]
    extents = {n: np.asarray(ref["obs_obj_ext"][0][i], np.float32) for i, n in enumerate(obj_names)}

    from vlm_base.sim_free_core import _A_Q01, _A_Q99, _load_droid_norm_stats
    loaded = _load_droid_norm_stats()
    a_q01, a_q99 = (loaded[0], loaded[1]) if loaded is not None else (_A_Q01, _A_Q99)

    with open("task_prompts.json", encoding="utf-8") as f:
        meta = json.load(f)[args.task]

    # init from a same-parameterization reference (Eq. 6), or train from scratch (x0 mode replaces the
    # score-reference init since the parameterizations differ).
    ckpt = torch.load(args.init, map_location="cpu") if args.init else None
    cfg = GeomProxyConfig(**ckpt["config"]) if ckpt else GeomProxyConfig(predict=args.predict)
    ds = GeomTaskDataset(args.demos, obj_names, extents, a_q01, a_q99, cfg.action_horizon,
                         meta["grasp_objs"], meta["place_obj"])
    print(f"[geom_proxy] {len(ds)} demo windows; predict={cfg.predict}; "
          f"{'init from ' + args.init if args.init else 'from scratch'}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeomScoreProxy(cfg).to(device)
    if ckpt:
        model.load_state_dict(ckpt["state_dict"])
    run = _wandb_init(args, cfg, "task")
    _run_training(model, ds, args, "task", device, run)
    _save(model, cfg, args.out, "task")
    if run is not None:
        run.finish()


def main():
    ap = argparse.ArgumentParser(description="Train geometry-conditioned score proxies (Config B)")
    ap.add_argument("--mode", choices=["reference", "task"], default="reference",
                    help="reference: distill the base score; task: score-match demos (init from reference)")
    ap.add_argument("--labels", type=str, default="data/weight_ref_labels/ref_s*.npz",
                    help="glob of the recorded reference-label .npz files (also gives object names/extents)")
    ap.add_argument("--demos", type=str, default=None, help="task mode: demo hdf5 to score-match")
    ap.add_argument("--init", type=str, default=None, help="task mode: reference model.pt to initialize from")
    ap.add_argument("--predict", choices=["score", "x0"], default="score",
                    help="task parameterization: x0 predicts the clean action (fixes the low-noise score blowup)")
    ap.add_argument("--task", type=str, default="weight",
                    help="task name for grasp/place roles (task_prompts.json)")
    ap.add_argument("--out", type=str, required=True, help="directory to write the trained proxy into")
    ap.add_argument("--epochs", type=int, default=200, help="passes over the training set")
    ap.add_argument("--batch_size", type=int, default=256, help="items per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    ap.add_argument("--log_every", type=int, default=10, help="epochs between loss prints")
    ap.add_argument("--wandb", action="store_true", help="log per-epoch loss to wandb (needs wandb login)")
    ap.add_argument("--wandb_project", type=str, default="pps", help="wandb project name")
    ap.add_argument("--wandb_entity", type=str, default=None,
                    help="wandb entity/team (default: your login's default)")
    ap.add_argument("--wandb_name", type=str, default=None, help="wandb run name (default: <role>-<task>)")
    args = ap.parse_args()
    (train_reference if args.mode == "reference" else train_task)(args)


if __name__ == "__main__":
    main()
