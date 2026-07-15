# Local MPC Proposal 与 PPS Score Compatibility

## 问题

MPC 不再使用原始 MBD proposal，而改为在当前 diffusion state 附近进行局部搜索：

\[
a_i = z_t + 0.8\sqrt{1-\bar\alpha_t}\,\epsilon_i,
\qquad \epsilon_i\sim\mathcal N(0,I).
\]

通过 cost weighting 得到 clean-action estimate \(\hat a_0\) 后，将其转换为 MPC score：

\[
s_{\mathrm{MPC}}(z_t,o,t)
=
\frac{\sqrt{\bar\alpha_t}\hat a_0-z_t}
{1-\bar\alpha_t}.
\]

核心问题是：这个修改后的 MPC score 能否在最终 evaluation 中与 task policy 和 reference policy 的 score 兼容，从而完成 PPS？

## 结论

**可以在工程和数值意义上兼容 PPS，但未必仍然是原始 diffusion distribution 的严格 probability score。**

上述转换使 \(s_{\mathrm{MPC}}\) 与 task/ref policy 输出具有相同的：

- tensor shape；
- normalized action space；
- timestep conditioning；
- score 量纲。

因此可以直接进行当前 eval 中的 score arithmetic：

\[
s_{\mathrm{combined}}
=
\gamma_{\mathrm{base}}s_{\mathrm{MPC}}
+\lambda\left(s_{\mathrm{task}}-s_{\mathrm{ref}}\right).
\]

当前实现位于：

- eval_steering.py：组合 base、task 和 ref scores；
- sim_free_mpc/planner.py：将 MPC 得到的 \(\hat a_0\) 转换为 score。

## PPS 能否成立的关键条件

真正关键的条件不是 proposal 是否严格来自 forward diffusion，而是 reference policy 必须在 eval 会访问的 states 上满足：

\[
s_{\mathrm{ref}}(z_t,o,t)
\approx
s_{\mathrm{MPC}}(z_t,o,t).
\]

当 \(\gamma_{\mathrm{base}}=\lambda=1\) 时：

\[
\begin{aligned}
s_{\mathrm{combined}}
&=s_{\mathrm{MPC}}+s_{\mathrm{task}}-s_{\mathrm{ref}}\\
&\approx s_{\mathrm{task}}.
\end{aligned}
\]

所以从 PPS 的 residual/cancellation 角度看，这个 local proposal 可以使用。即使 \(s_{\mathrm{MPC}}\) 不是某个严格可积 density 的精确 score，只要 ref policy 能 pointwise mimic 同一个 MPC vector field，score cancellation 仍然能够工作。

## 必须保持一致的设置

生成 ref cache 和最终 eval 应使用完全一致的：

- proposal center 和 radius：
  \[
  z_t+0.8\sqrt{1-\bar\alpha_t}\epsilon;
  \]
- alpha/beta schedule；
- diffusion train timesteps 和 reverse steps；
- timestep convention；
- action normalization statistics；
- action horizon 和 action dimensions；
- MPC cost function；
- temperature；
- candidate count；
- optimizer iterations；
- constraints、clipping 和 interpolation 设置。

task/ref policy 都必须输出 score。不能将 score、epsilon prediction 和 velocity prediction 直接混合。

此外，ref cache 中的 \(z_t\) 应来自与 eval 一致的、从 Gaussian noise 开始的 reverse denoising trajectories，从而覆盖 inference 实际访问的 state distribution。

## 最大的实际风险：有限样本噪声

ref policy 通常学到的是对 MPC sampling randomness 平均后的结果：

\[
s_{\mathrm{ref}}
\approx
\mathbb E_{\text{MPC randomness}}
\left[s_{\mathrm{MPC}}\right].
\]

但 eval 中的 \(s_{\mathrm{MPC}}\) 是有限候选数下的一次随机估计。如果 \(N=512\) 时仍有明显噪声，那么

\[
s_{\mathrm{MPC}}-s_{\mathrm{ref}}
\]

不会完全抵消，而会在 combined score 中留下 sampling noise。这个问题比 proposal 是否保持严格的 diffusion analogy 更可能直接影响 closed-loop PPS performance。

## 与 importance sampling 意见的关系

如果希望将修改后的估计严格解释为原始 diffusion target 的 score，那么改变 proposal 后原则上需要使用 importance weights：

\[
w_i\propto
\exp\left[-J(a_i,o)/\lambda_T\right]
\frac{p_{\mathrm{diff}}(a_i\mid z_t)}
{q(a_i\mid z_t)},
\]

其中

\[
q(a_i\mid z_t)
=
\mathcal N\!\left(
z_t,\;0.8^2(1-\bar\alpha_t)I
\right).
\]

因此，对“修改 proposal 会改变严格 diffusion interpretation”的担忧是成立的。但 importance sampling 主要决定是否还能声称输出是原始 target distribution 的严格 score；它不是 score arithmetic 或 PPS cancellation 能够运行的必要条件。在高维 action horizon 中，density ratio 还可能造成 weight degeneracy，需要单独验证。

## 建议的兼容性实验

### 1. Pointwise matching

在 held-out reverse trajectories 上，按 timestep 分别计算：

\[
\operatorname{cos}
\left(s_{\mathrm{MPC}},s_{\mathrm{ref}}\right)
\]

和

\[
\frac{
\left\|s_{\mathrm{MPC}}-s_{\mathrm{ref}}\right\|
}{
\left\|s_{\mathrm{MPC}}\right\|+\varepsilon
}.
\]

不能只观察整体 training MSE，因为 early/late diffusion timesteps 的 score scale 和误差可能差异很大。

### 2. Null steering test

临时令

\[
s_{\mathrm{task}}=s_{\mathrm{ref}}.
\]

打开 steering 后，结果应与纯 MPC base 基本一致。该实验检查 score combination 和 update implementation 是否正确。

### 3. Cancellation test

设置

\[
\gamma_{\mathrm{base}}=\lambda=1,
\]

比较

\[
s_{\mathrm{MPC}}+s_{\mathrm{task}}-s_{\mathrm{ref}}
\]

与 \(s_{\mathrm{task}}\) 的 cosine similarity 和 norm ratio。如果 ref distillation 足够准确，两者应非常接近。

## 最终判断

local trust-region proposal

\[
a_i=z_t+0.8\sqrt{1-\bar\alpha_t}\epsilon_i
\]

能够和 task/ref score policies 一起完成工程意义上的 PPS。成败主要取决于 reference policy 是否准确蒸馏了**使用同一个 local proposal 和同一套 MPC 配置产生的 score field**，尤其是在 eval reverse trajectories 实际访问的 states 上。

importance sampling 影响的是严格概率解释，而不是 PPS score subtraction 的基本数值兼容性。实际评估中应优先检查 ref/base pointwise matching、不同 timestep 的误差，以及有限样本 MPC noise 是否会在 cancellation 后残留。
