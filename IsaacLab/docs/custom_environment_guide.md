# Custom Environment Creation Guide

This guide explains how to create custom environments with different lighting and mass properties for the Isaac Lab Mimic stack cube task.

## Files Created

### 1. Main Custom Environment Configuration
**File**: `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_visuomotor_custom_mimic_env_cfg.py`

This file contains the main custom environment configuration that inherits from the base visuomotor environment.

### 2. Custom Mass Properties Environment (Advanced)
**File**: `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_visuomotor_custom_mass_mimic_env_cfg.py`

This file demonstrates how to modify cube mass and friction properties (more complex implementation).

### 3. Environment Registration
**File**: `source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py` (modified)

Added registration for the new custom environment:
```python
gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_custom_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCustomMimicEnvCfg,
    },
    disable_env_checker=True,
)
```

### 4. Test Script
**File**: `scripts/imitation_learning/isaaclab_mimic/test_custom_environments.py`

A script to test the custom environments and compare them with the original.

## How to Use

### 1. Basic Usage
```python
import gymnasium as gym

# Create the custom environment
env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Mimic-v0")
obs, _ = env.reset()
# ... use the environment
env.close()
```

### 2. Test the Environments
```bash
# Test all custom environments
python scripts/imitation_learning/isaaclab_mimic/test_custom_environments.py

# Test specific environment
python scripts/imitation_learning/isaaclab_mimic/test_custom_environments.py --env_name "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Mimic-v0"

# Run in headless mode
python scripts/imitation_learning/isaaclab_mimic/test_custom_environments.py --headless
```

## Customization Options

### Lighting Modifications
To modify lighting properties, you can:

1. **Change base lighting**: Modify the `light` configuration in the scene
2. **Modify lighting randomization**: Override the `randomize_light` event parameters
3. **Add custom lighting events**: Create new event configurations

Example lighting modifications:
```python
# In your custom environment configuration
self.events.randomize_light.params.update({
    "intensity_range": (500.0, 5000.0),  # Different intensity range
    "color_variation": 0.6,  # More color variation
    "default_intensity": 2000.0,  # Lower default intensity
    "default_color": (0.9, 0.9, 1.0),  # Slightly blue tint
})
```

### Mass and Physics Modifications
To modify mass and friction properties:

1. **Create custom USD files**: Modify the cube USD files with different mass properties
2. **Override cube configurations**: Recreate cube configurations with custom properties
3. **Add physics material randomization**: Create events that randomize material properties

Example mass modifications:
```python
# Custom physics material
custom_physics_material = RigidBodyMaterialCfg(
    static_friction=0.3,  # Lower static friction
    dynamic_friction=0.3,  # Lower dynamic friction
    restitution=0.1,  # Slight bounce
    friction_combine_mode="multiply",
)

# Custom mass properties
custom_mass_props = MassPropertiesCfg(
    mass=0.8,  # Lighter cubes
)
```

## Environment Hierarchy

The custom environment inherits from:
```
FrankaCubeStackIKRelVisuomotorCustomMimicEnvCfg
├── FrankaCubeStackVisuomotorEnvCfg
│   ├── FrankaCubeStackEnvCfg
│   │   └── StackEnvCfg
│   │       └── ManagerBasedRLEnvCfg
└── MimicEnvCfg
```

## Key Configuration Parameters

### Lighting Parameters
- `intensity_range`: Range for light intensity randomization
- `color_variation`: Amount of color variation (0.0 to 1.0)
- `default_intensity`: Base light intensity
- `default_color`: Base light color (RGB tuple)
- `textures`: List of HDR texture paths for environment lighting

### Mass and Physics Parameters
- `mass`: Explicit mass value in kg
- `density`: Material density in kg/m³
- `static_friction`: Static friction coefficient
- `dynamic_friction`: Dynamic friction coefficient
- `restitution`: Bounce/restitution coefficient
- `friction_combine_mode`: How friction is combined during collisions
- `restitution_combine_mode`: How restitution is combined during collisions

### Datagen Parameters
- `name`: Dataset name for data generation
- `generation_guarantee`: Whether to guarantee successful data generation
- `generation_keep_failed`: Whether to keep failed generation attempts
- `generation_num_trials`: Number of trials for generation
- `max_num_failures`: Maximum number of failures allowed

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required modules are imported correctly
2. **Configuration Conflicts**: Ensure custom configurations don't conflict with parent classes
3. **USD File Issues**: Verify USD file paths are correct and files exist
4. **Physics Simulation**: Check that physics parameters are within valid ranges

### Debugging Tips

1. **Use the test script**: Run the test script to verify environment creation
2. **Check logs**: Look for error messages in the simulation logs
3. **Verify inheritance**: Ensure proper inheritance from parent configuration classes
4. **Test incrementally**: Start with minimal changes and add complexity gradually

## Next Steps

1. **Create more variants**: Add more custom environments with different parameter combinations
2. **Add domain randomization**: Implement more sophisticated randomization strategies
3. **Performance optimization**: Optimize environment configurations for better performance
4. **Documentation**: Add more detailed documentation for specific use cases

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab Mimic Documentation](https://isaac-sim.github.io/IsaacLab/source/setup/isaaclab_mimic.html)
- [USD Physics Documentation](https://openusd.org/release/wp_rigid_body_physics.html)



