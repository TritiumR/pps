# ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
#     --enable_cameras \
#     --num_envs 1 \
#     --generation_num_trials 50 \
#     --task Isaac-Weight-Droid-IK-Rel-Visuomotor-Mimic-v0 \
#     --input_file ../diffusion_policy/data/weight/annotated_dataset.hdf5 \
#     --output_file ../diffusion_policy/data/weight/generated_dataset.hdf5 \
#     --seed 42 \
#     --headless

# ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
#     --enable_cameras \
#     --num_envs 1 \
#     --generation_num_trials 50 \
#     --task Isaac-Pot-Droid-IK-Rel-Visuomotor-Mimic-v0 \
#     --input_file ../diffusion_policy/data/pot/annotated_dataset.hdf5 \
#     --output_file ../diffusion_policy/data/pot/generated_dataset.hdf5 \
#     --seed 42 \
#     --headless

# ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
#     --enable_cameras \
#     --num_envs 1 \
#     --generation_num_trials 50 \
#     --task Isaac-Drink-Droid-IK-Rel-Visuomotor-Mimic-v0 \
#     --input_file ../diffusion_policy/data/drink/annotated_dataset.hdf5 \
#     --output_file ../diffusion_policy/data/drink/generated_dataset.hdf5 \
#     --seed 42 \
#     --headless
    
# ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
#     --enable_cameras \
#     --num_envs 1 \
#     --generation_num_trials 50 \
#     --task Isaac-Can-Droid-IK-Rel-Visuomotor-Mimic-v0 \
#     --input_file ../diffusion_policy/data/can/annotated_dataset.hdf5 \
#     --output_file ../diffusion_policy/data/can/generated_dataset.hdf5 \
#     --seed 42 \
#     --headless

# ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
#     --enable_cameras \
#     --num_envs 1 \
#     --generation_num_trials 50 \
#     --task Isaac-Pen-Droid-IK-Rel-Visuomotor-Mimic-v0 \
#     --input_file ../diffusion_policy/data/pen/annotated_dataset.hdf5 \
#     --output_file ../diffusion_policy/data/pen/generated_dataset.hdf5 \
#     --seed 42 \
#     --headless


# ./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
#     --input_file ../diffusion_policy/data/pot/generated_dataset.hdf5 \
#     --output_dir ../diffusion_policy/data/pot/videos \
#     --video_height 180 \
#     --video_width 320

# ./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
#     --input_file ../diffusion_policy/data/weight/generated_dataset.hdf5 \
#     --output_dir ../diffusion_policy/data/weight/videos \
#     --video_height 180 \
#     --video_width 320

./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
    --input_file ../diffusion_policy/data/drink/generated_dataset.hdf5 \
    --output_dir ../diffusion_policy/data/drink/videos \
    --video_height 180 \
    --video_width 320

# ./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
#     --input_file ../diffusion_policy/data/can/generated_dataset.hdf5 \
#     --output_dir ../diffusion_policy/data/can/videos \
#     --video_height 180 \
#     --video_width 320

./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
    --input_file ../diffusion_policy/data/pen/generated_dataset.hdf5 \
    --output_dir ../diffusion_policy/data/pen/videos \
    --video_height 180 \
    --video_width 320

uv run examples/Isaaclab/convert_isaaclab_data_to_lerobot.py \
    --data_file ../diffusion_policy/data/weight/generated_dataset.hdf5 \
    --repo_name "cn356/isaaclab_weight" \
    --prompt "remove the lid of the pot and put egg in it"