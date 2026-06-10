#!/bin/bash
# Workaround for libstdc++ compatibility issue

# Unset conda library paths to use system libraries
export LD_LIBRARY_PATH=""
export LD_PRELOAD=""

# Run with system python (which should have all deps installed)
/home/chuanruo/IsaacLab/isaaclab.sh --python scripts/tools/render_env_video_simple.py --task Isaac-Laptop-Droid-Visuomotor-v0 --num_steps 200 --headless "$@"
