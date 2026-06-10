## Overview

Isaac Lab is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim. It provides a unified interface for reinforcement learning, imitation learning, and motion planning. The framework supports two environment design workflows:

- **Manager-based**: Decomposes environments into configurable managers (observations, actions, rewards, terminations, events, curriculum)
- **Direct**: Single-class implementation where users implement all functionality directly

## Common Commands

All development commands are handled through the `./isaaclab.sh` (Linux) or `isaaclab.bat` (Windows) script.

### Environment Setup

```bash
# Create conda environment (default name: env_isaaclab)
./isaaclab.sh --conda [env_name]

# Create uv environment
./isaaclab.sh --uv [env_name]

# Install Isaac Lab extensions and RL frameworks
./isaaclab.sh --install [all|none|rsl_rl|rl_games|sb3|skrl]
```

### Running Code

```bash
# Run Python with Isaac Sim environment
./isaaclab.sh --python <script.py>

```

# To check usd
Use the python in /home/chuanruo/Downloads/blender-4.3.2-linux-x64/4.3/python/bin/python3.11

### Testing and Linting

### Documentation

```bash
# Build documentation
./isaaclab.sh --docs

# Open built documentation
xdg-open docs/_build/current/index.html
```

### Docker

```bash
# Build and run Docker container
python3 docker/container.py
```

### Project Templates

```bash
# Create new task or project from template
./isaaclab.sh --new
```

## Architecture

### Package Structure

The repository is organized as a monorepo with multiple Python packages:

- **`source/isaaclab/`**: Core framework providing simulation interface, environments, sensors, assets, and managers
- **`source/isaaclab_assets/`**: Robot and object asset configurations
- **`source/isaaclab_tasks/`**: Task implementations (manager-based and direct workflows)
- **`source/isaaclab_rl/`**: RL framework integrations (RSL-RL, RL-Games, Stable-Baselines3, SKRL)
- **`source/isaaclab_mimic/`**: Imitation learning support (Apache 2.0 licensed)

### Manager-Based Workflow

Manager-based environments use a configuration-driven approach with these key managers:

- **ActionManager**: Applies actions to assets (joint positions, joint velocities, inverse kinematics)
- **ObservationManager**: Computes observations from scene entities
- **RewardManager**: Computes reward signals
- **TerminationManager**: Determines episode termination conditions
- **EventManager**: Handles domain randomization and scene resets
- **CommandManager**: Generates high-level commands for the robot
- **CurriculumManager**: Manages curriculum learning

Configuration classes (e.g., `ObservationTermCfg`, `RewardTermCfg`) specify:
- `func`: The function to call
- `weight`: Weight for rewards/observations
- `params`: Parameters passed to the function

### Environment Configuration Pattern

Environment configurations use a hierarchical `@configclass` pattern:

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

Tasks are registered via Gymnasium entry points in `source/isaaclab_tasks/config/extension.toml`.

### Key Directories for Tasks

- **`source/isaaclab_tasks/isaaclab_tasks/manager_based/`**: Manager-based task implementations
  - `manipulation/`, `locomotion/`, `navigation/`, `classic/`
  - Each task has `mdp/` subdirectory with reward/observation/termination functions
  - `config/` subdirectory with environment configurations
- **`source/isaaclab_tasks/isaaclab_tasks/direct/`**: Direct workflow implementations

### Asset Configuration

Assets are configured using dataclasses in `isaaclab.assets`:

- **`ArticulationCfg`**: Robots with joints
- **`RigidObjectCfg`**: Static or dynamic rigid objects
- **`DeformableObjectCfg`**: Deformable objects
- **`AssetBaseCfg`**: Base class for all assets

Spawn configurations control how assets are loaded:
- `UsdFileCfg`: Load from USD file
- `MjcfFileCfg`: Load from MJCF/URDF
- `SphereCfg`, `CapsuleCfg`, `CuboidCfg`: Procedural primitives

### Sensor System

Sensors are configured and attached to scene entities:

- **`CameraCfg`**: RGB/depth/segmentation cameras (RTX-based)
- **`RayCasterCfg`**: Ray-casting for height maps or distance sensors
- **`ContactSensorCfg`**: Contact force/torque sensing
- **`ImuCfg`**: Inertial measurement units
- **`FrameTransformerCfg`**: Transform tracking between frames

### RL Training Scripts

Training scripts follow a consistent pattern across RL frameworks:

1. Parse arguments with framework-specific CLI args
2. Launch Isaac Sim via `AppLauncher`
3. Import and instantiate environment
4. Configure and run training

Each framework has `train.py` and `play.py` in `scripts/reinforcement_learning/<framework>/`.

## Code Style

- **Line length**: 120 characters
- **Formatter**: Black with `--unstable` flag
- **Import sorting**: isort with custom sections (see `pyproject.toml`)
- **Linting**: flake8 with specific ignores (E402, E501, W503, E203, D401, R504, R505, SIM102, SIM117, SIM118)
- **Type checking**: Pyright (basic mode, missing imports ignored in CI)
- **License headers**: BSD-3 for most files, Apache 2.0 for `isaaclab_mimic`

## Important File Patterns

- `**/mdp/*.py`: Markov Decision Process functions (rewards, observations, terminations)
- `**/config/*.py`: Environment and agent configurations
- `**/agents/*.py`: RL algorithm configurations
- `**/*_env_cfg.py`: Environment configuration classes
- `**/*_cfg.py`: General configuration classes

## Environment Variables

- `ISAACLAB_PATH`: Root of the repository
- `ISAAC_PATH`: Isaac Sim installation directory (set by `_isaac_sim/setup_conda_env.sh`)
- `CARB_APP_PATH`, `EXP_PATH`: Isaac Sim internal paths


## Mimic Demo Deneration

### Collect demos

```bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Env_Name-IK-Rel-v0 --teleop_device oculus --dataset_file Demos_path --num_demos Number_of_demos --enable_cameras
```

### Annotate demos

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
--enable_cameras --task Env_Name-Mimic-v0 --auto \
--input_file Demos_path --output_file Annotated_demo_path --headless
```

### Generate demos

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \        
    --enable_cameras \
    --num_envs 1 \
    --generation_num_trials Number_of_generated_demos \
    --task Env_Name-Mimic-v0 \
    --input_file Annotated_demo_path \
    --output_file Generated_demo_path \
    --seed SEED \
    --headless
```
