#!/bin/bash
# Wrapper script to fix libstdc++ compatibility issue

# Save original LD_LIBRARY_PATH
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Remove conda lib paths to use system libstdc++
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "anaconda3\|conda" | tr '\n' ':')

# If empty, set to system default
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
fi

# Run the actual command
exec /home/chuanruo/IsaacLab/isaaclab.sh --python scripts/tools/render_env_video_simple.py "$@"
