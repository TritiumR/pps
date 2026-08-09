# ReKep 工作进度

最后更新：2026-08-09

## 当前状态

- 当前范围只包含 Weight、Capsule、Pot；Tea 暂停，Task-only 不再运行。
- GroundingDINO+SAM 的 pre-ReKep cache 已生成并逐 seed 通过 preflight 审计。缓存只固定感知前端，ReKep keypoint proposal、grounding、规划和 manipulation logic 每次 eval 仍重新执行。
- 缓存根目录：`/autodl-fs/data/yl4535/pps/perception_cache/groundedsam_pre_rekep`。
- 所有正式 eval 统一使用 `--determine --percept_cache`。
- 通过审计的正式 seed：

  - Weight：1–20。
  - Capsule：2, 4, 5, 8, 15, 18, 22, 25, 27, 28, 29, 32, 36, 41, 42, 43, 44, 45, 46, 48。
  - Pot：1, 2, 4, 7, 9, 10, 13, 14, 15, 17, 18, 22, 25, 29, 31, 32, 34, 38, 44, 45；46、47 也通过但不进入正式 20 seeds。
- GPU 0–2 已分别启动 Weight、Capsule、Pot 的串行队列：Base → steer 0.4 → 0.2 → 0.6 → 0.8。
- GPU 3 专门探索 Capsule/Pot manipulation logic，不与正式结果混用。

## 待办

- [ ] 完成 Weight、Capsule、Pot 的 cached deterministic Base。
- [ ] 完成三个任务 steering scale 0.2、0.4、0.6、0.8 的 sweep，并按 task/seed 汇总成功率、失败阶段和代表视频。
- [ ] 完成 Capsule manipulation 优化：排除 gripper 遮挡 lid 的起始场景；让工具在 lid lip 处侧倾并沿铰链方向上撬。
- [ ] 完成 Pot manipulation 优化：稳定定位 lid handle、适配细把手的 aperture 判定，并在移盖成功后及时释放转入 egg 阶段。
- [ ] 在至少两个通过 cache 审计的 seed 上复核有效修改，再决定是否合并 manipulation 分支。

## GPU 分工与实验约束

- GPU 0：Weight 正式队列。
- GPU 1：Capsule 正式队列。
- GPU 2：Pot 正式队列。
- GPU 3：Capsule/Pot pilot。
- Tea 和 Task-only 均不运行。
- 正式队列使用稳定分支 `vlm-dp-clean-isaaclab2.3`；操作逻辑实验使用 `vlm-dp-clean-isaaclab2.3-manip`，避免污染 baseline。

## 操作逻辑审计结论

- `raw.txt` 描述阶段、keypoint 约束和语义目标，但不会直接输出机械臂动作。
- `fake_vlm.py`/ReKep compiler 选择 grasp keypoint、contact mode、target 和 stage；cost terms 把这些约束变成 MPC 代价；`bridge.py` 根据距离、aperture 和 hold 状态推进或回退阶段。
- Pot 旧逻辑抓 cover 的整体中心，容易下压 lid；当前 pilot 已改为 raw cover mask 上凸起的 handle 点，并声明局部轴和宽度。
- Pot handle 约 10 mm 宽，固定 `stall_margin=0.15` 会把真实细把手抓取误判成 closed-empty；实验分支已加入局部宽度自适应阈值。
- Capsule 成功 task-only episode 的关键动作是腕部侧倾；实验分支已支持 plan 声明 tool axis 和 stage-local orientation authority。

## Weight Eval 命令模板

核对说明：

- Base 和 Steer 中裸 `--percept_cache` 默认读取当前缓存根目录；Weight cache 生成完成后才能运行。
- 当前计划不运行 Task-only。
- Steer 显式使用 `--task_attention bidirectional`。

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
