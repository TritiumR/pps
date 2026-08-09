# ReKep 工作进度

最后更新：2026-08-08

## 当前状态

- Tea、Capsule 和 Pot 的 seed 1–20 确定性感知缓存已经生成完毕。
- 缓存根目录：

  ```text
  /autodl-fs/data/yl4535/pps/perception_cache/groundedsam_pre_rekep/
  ├── tea/seed_0001 ... seed_0020
  ├── capsule/seed_0001 ... seed_0020
  └── pot/seed_0001 ... seed_0020
  ```

- 每个 seed 目录包含：

  - `perception.npz`：RGB、世界坐标点云、命名 mask 和 label image。
  - `metadata.json`：prompt、GroundingDINO 检测框及置信度、最终框、mask 大小和 workspace bounds。
  - `visualization.png`：叠加 SAM mask、检测框、物体名称、置信度以及缺失 mask 提示的可视化。

- Eval 使用缓存的参数：

  ```bash
  --percept_cache /autodl-fs/data/yl4535/pps/perception_cache/groundedsam_pre_rekep
  ```

- 缓存只跳过 GroundingDINO+SAM；ReKep keypoint proposal 以及后续 grounding、规划和 manipulation logic 每次 eval 仍会重新执行。
- `--percept_cache` 必须和 `--determine` 一起使用，保证 seed 对应同一个确定性初始场景。
- 之前启动的四个 task-only eval 已停止，部分结果暂时保留。

## 待办

- [ ] 使用 cache + determine 重跑 Task Policy、Weight Base 和 Weight Steer。

  - 四个 Task-only policy：seed 1–20，只使用 `--determine`。Task-only 不经过 VLM/ReKep 感知，因此不使用 cache。
  - Weight Base 和 Weight Steer：先生成并检查 Weight seed 1–20 感知缓存，再使用 `--percept_cache` 和 `--determine` 评估。

- [ ] 弄清楚并优化 manipulation logic。

  首先只做代码路径审计，确认以下问题，再决定如何修改：

  - 什么逻辑决定夹爪何时张开、闭合？
  - 什么逻辑决定机械臂靠近哪个 keypoint？
  - `raw.txt` 是否直接决定这些行为，还是只提供 keypoint 约束？
  - stage 构造、cost terms、gripper gates 和 bridge 状态转换分别承担什么作用？
  - 从 ReKep constraint/keypoint claim 到实际 approach target 和 gripper command 的完整调用链是什么？

- [ ] 运行其余任务的 Base + Steer。

  - 任务范围：Tea、Capsule、Pot。
  - Tea 和 Capsule 使用已经完成的 seed 1–20 cache。
  - Pot 使用已经完成的 seed 1–20 cache。
  - 所有评估统一使用 `--determine`。

- [ ] 对所有任务 sweep steering scale，并总结不同 case。

  - 任务范围：Weight、Tea、Pot、Capsule。
  - Scale：`0.2`、`0.4`、`0.6`、`0.8`。
  - 记录每个 task/seed 的成功状态、失败阶段、感知状态和代表性视频。
  - 与对应的 cached deterministic Base 和 Task-only baseline 对比。

## Weight Eval 命令模板

核对说明：

- Base 和 Steer 中裸 `--percept_cache` 默认读取当前缓存根目录；Weight cache 生成完成后才能运行。
- Task-only 不经过 VLM/ReKep，因此只使用 `--determine`，不能加 `--percept_cache`。
- Task-only 和 Steer 都显式使用 `--task_attention bidirectional`。当前训练配置默认值仍是 causal，不能依赖配置默认值。
- Task-only 模板中的 `--gpus 2` 表示使用 GPU 2；按实际空闲卡修改。

### Base policy

```bash
python eval_steering.py \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --vlm_base --no_steer --base_decode_only \
  --vlm_cost rekep_fake --vlm_state real --vlm_track visual --vlm_segment groundedsam \
  --vlm_cost_config vlm_dp/configs/test_configs/parity36_calm_rise5.yaml \
  --mpc_update mbd_score_action_prox --mpc_cost priority --mpc_optimize_space action \
  --mpc_num_samples 4096 --mpc_iterations 1 --mpc_noise 0.4 --mpc_temperature 0.1 \
  --mpc_joint_delta_clip 0.05 --num_steps 10 --mpc_ddim_train_timesteps 100 \
  --cost_executable_actions \
  --task_num_steps 1600 --steps_per_inference 4 --interpolate --headless --mpc_debug \
  --determine --percept_cache --video_stride 4 --device cuda:0 \
  --exp_name rebase_repro --seed_start 1 --seed_end 21
```

### Task policy

```bash
python eval_steering.py \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --task_only --task_attention bidirectional \
  --task_checkpoint_dir openpi/checkpoints/score_task_weight/task_eps_bidir_openpi_image_only/30000 \
  --num_steps 10 --task_num_steps 1600 \
  --steps_per_inference 4 \
  --seed_start 1 --seed_end 21 \
  --workers 1 --gpus 2 \
  --task_debug --determine \
  --exp_name task_only_1600
```

### Steered policy（scale 0.4）

```bash
python eval_steering.py \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --task_steer --task_attention bidirectional --base_decode_only \
  --task_checkpoint_dir openpi/checkpoints/score_task_weight/task_eps_bidir_openpi_image_only/30000 \
  --steer_scale 0.4 \
  --vlm_cost rekep_fake --vlm_state real --vlm_track visual --vlm_segment groundedsam \
  --vlm_cost_config vlm_dp/configs/test_configs/parity36_calm_rise5.yaml \
  --mpc_update mbd_score_action_prox --mpc_cost priority --mpc_optimize_space action \
  --mpc_num_samples 4096 --mpc_iterations 1 --mpc_noise 0.4 --mpc_temperature 0.1 \
  --mpc_joint_delta_clip 0.05 --num_steps 10 --mpc_ddim_train_timesteps 100 \
  --cost_executable_actions \
  --task_num_steps 1600 --steps_per_inference 4 --interpolate --headless --mpc_debug \
  --determine --percept_cache --video_stride 4 --device cuda:0 \
  --exp_name vlm_task_steer_s04 \
  --seed_start 1 --seed_end 21
```
