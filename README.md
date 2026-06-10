## Run evaluation in IsaacLab

```bash
sh test_isaac.sh
```

## Run steering in IsaacLab

```bash
sh test_steer_separate.sh
```


## Run pi-0.5 on droid

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=checkpoints/pi05_droid
```