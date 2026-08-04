# Bimanual Concerto point-cloud prefix

`BimanualProxyConfig` keeps the three DINOv3 RGB streams and adds one point
cloud for the scene camera and each wrist camera. All clouds use one shared,
frozen `concerto_small` encoder. A trainable Perceiver resampler produces 128
tokens per camera, so the Gemma prefix receives 384 point-cloud tokens in
addition to the DINO image tokens.

## Runtime dependency

Install the official [Pointcept/Concerto](https://github.com/Pointcept/Concerto)
package in the OpenPI training environment. Match the SpConv and torch-scatter
wheels to the environment's CUDA and PyTorch versions:

```bash
pip install spconv-cu${CUDA_VERSION}
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install huggingface_hub timm addict
git clone https://github.com/Pointcept/Concerto.git
pip install -e ./Concerto
```

FlashAttention is optional. The integration defaults to
`concerto_enable_flash=False`; enable it in the model config only when the
official `flash-attn` package is installed.

The first model construction downloads `concerto_small.pth` from
`Pointcept/Concerto` on Hugging Face. Set `concerto_checkpoint_dir` to choose
the cache directory.

## LeRobot sample fields

The bimanual data factory expects three XYZ/RGB pairs, each shaped `(N, 3)`:

```text
point_position
point_color
left_wrist_point_position
left_wrist_point_color
right_wrist_point_position
right_wrist_point_color
```

It maps them to the canonical model keys:

```text
base_0_pointcloud
left_wrist_0_pointcloud
right_wrist_0_pointcloud
```

Use `BimanualProxyConfig` together with
`BimanualProxyLeRobotDROIDJointPosDataConfig`. `pointcloud_num_points` controls
the deterministic input resize and defaults to 1024; it does not change the
fixed 128-token output per camera.
