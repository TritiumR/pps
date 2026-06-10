# CLAUDE.md

Brief reference for Isaac Lab — GPU-accelerated robotics simulation on NVIDIA Isaac Sim. See `AGENTS.md` for full details.

## Workflows

- **Manager-based**: Config-driven via managers (Action, Observation, Reward, Termination, Event, Command, Curriculum).
- **Direct**: Single-class implementation.

## Common Commands

All commands run through `./isaaclab.sh` (Linux) or `isaaclab.bat` (Windows).

```bash
./isaaclab.sh --conda [env_name]            # Create conda env (default: env_isaaclab)
./isaaclab.sh --install [all|rsl_rl|...]    # Install extensions + RL frameworks
./isaaclab.sh --python <script.py>          # Run script in Isaac Sim env
./isaaclab.sh --docs                        # Build docs
./isaaclab.sh --new                         # Scaffold new task/project
python3 docker/container.py                 # Docker build/run
```

USD inspection uses `/home/chuanruo/Downloads/blender-4.3.2-linux-x64/4.3/python/bin/python3.11`.

## Package Layout (`source/`)

- `isaaclab/` — core sim, envs, sensors, assets, managers
- `isaaclab_assets/` — robot/object configs
- `isaaclab_tasks/` — task implementations (manager-based + direct)
- `isaaclab_rl/` — RSL-RL, RL-Games, SB3, SKRL integrations
- `isaaclab_mimic/` — imitation learning (Apache 2.0)

## Manager-Based Env Pattern

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
```

Term configs (`ObservationTermCfg`, `RewardTermCfg`, etc.) take `func`, `weight`, `params`. Tasks register via Gymnasium entry points in `source/isaaclab_tasks/config/extension.toml`.

## Key File Patterns

- `**/mdp/*.py` — reward/observation/termination functions
- `**/*_env_cfg.py` — env configs
- `**/config/*.py`, `**/agents/*.py` — env + RL agent configs

## Assets & Sensors

- Assets: `ArticulationCfg`, `RigidObjectCfg`, `DeformableObjectCfg` with spawn cfgs (`UsdFileCfg`, `MjcfFileCfg`, primitive shapes).
- Sensors: `CameraCfg`, `RayCasterCfg`, `ContactSensorCfg`, `ImuCfg`, `FrameTransformerCfg`.

## RL Training

`scripts/reinforcement_learning/<framework>/{train,play}.py` — parse args, launch `AppLauncher`, instantiate env, train.

## Code Style

- Line length 120; Black `--unstable`; isort; flake8 (ignores E402/E501/W503/E203/D401/R504/R505/SIM102/SIM117/SIM118); Pyright basic.
- License: BSD-3 (Apache 2.0 for `isaaclab_mimic`).

## Env Vars

- `ISAACLAB_PATH` — repo root
- `ISAAC_PATH` — Isaac Sim install
- `CARB_APP_PATH`, `EXP_PATH` — Isaac Sim internals

## Mimic Demo Pipeline

```bash
# Collect
./isaaclab.sh -p scripts/tools/record_demos.py --task <Env>-IK-Rel-v0 \
  --teleop_device oculus --dataset_file <path> --num_demos N --enable_cameras

# Annotate
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --enable_cameras --task <Env>-Mimic-v0 --auto \
  --input_file <demos> --output_file <annotated> --headless

# Generate
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --enable_cameras --num_envs 1 --generation_num_trials N \
  --task <Env>-Mimic-v0 --input_file <annotated> --output_file <generated> \
  --seed SEED --headless
```
