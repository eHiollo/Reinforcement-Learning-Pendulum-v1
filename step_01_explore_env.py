"""
第一步：认识倒立摆环境

目标：搞懂「状态」「动作」「奖励」分别是什么，
      并能用 reset / step 和环境交互几步。

运行：python step_01_explore_env.py
"""

import gymnasium as gym
import numpy as np

# -----------------------------------------------------------------------------
# 1. 创建环境
# -----------------------------------------------------------------------------
# Gymnasium 里倒立摆叫 "Pendulum-v1"
# 注意：这个环境用的不是 MuJoCo，是经典物理公式。等我们写好流程，再换成 MuJoCo 版本也可以。
env = gym.make("Pendulum-v1")

print("=" * 50)
print("第一步：认识倒立摆环境")
print("=" * 50)
print()
print("【观察空间】observation_space（也就是我们说的「状态」）")
print("  ", env.observation_space)
print("  含义：每个 step 环境会给我们一个 3 维向量")
print("  - 第 0 维：cos(θ)，摆角 θ 的余弦")
print("  - 第 1 维：sin(θ)，摆角 θ 的正弦")
print("  - 第 2 维：θ 的角速度 θ_dot")
print("  当摆竖直向上时，θ=0，cos(θ)=1, sin(θ)=0。我们想让摆保持这个状态。")
print()
print("【动作空间】action_space")
print("  ", env.action_space)
print("  含义：我们每步可以选一个 1 维连续动作，表示施加的力矩，范围 [-2, 2]")
print()

# -----------------------------------------------------------------------------
# 2. 重置环境，拿到初始状态
# -----------------------------------------------------------------------------
obs, info = env.reset(seed=42)
# 习惯上我们把 observation 叫 state，在本项目里混用也可以
state = np.array(obs, dtype=np.float32)

print("【重置环境】reset(seed=42)")
print("  初始状态 state =", state)
print("  state[0] = cos(θ) =", state[0])
print("  state[1] = sin(θ) =", state[1])
print("  state[2] = θ_dot  =", state[2])
print()

# -----------------------------------------------------------------------------
# 3. 随便选一个动作，执行一步
# -----------------------------------------------------------------------------
# 动作是 1 维的，范围 [-2, 2]。这里我们选 0（不施力）做个演示
action = np.array([0.0], dtype=np.float32)

next_obs, reward, terminated, truncated, info = env.step(action)
next_state = np.array(next_obs, dtype=np.float32)
done = terminated or truncated

print("【执行一步】step(action=[0.0]) 即不施力")
print("  下一个状态 next_state =", next_state)
print("  本步奖励 reward       =", reward)
print("  是否结束 done         =", done)
print("  info                  =", info)
print()
print("  补充：Pendulum 的 reward 和「角度、角速度」有关，越接近竖直、角速度越小，reward 越大（越接近 0）。")
print()

# -----------------------------------------------------------------------------
# 4. 多跑几步，感受「状态 → 动作 → 新状态 → 奖励」的循环
# -----------------------------------------------------------------------------
print("【多跑几步】用随机动作跑 5 步，观察状态和奖励变化")
print()

env.reset(seed=123)
total_reward = 0.0

for t in range(5):
    # 随机选一个动作（仅作演示，之后会换成「策略」来选）
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    total_reward += reward
    state = np.array(next_obs, dtype=np.float32)

    print(f"  步数 t={t+1}: action={action[0]:7.3f}  reward={reward:7.3f}  "
          f"cos(θ)={state[0]:7.3f}  sin(θ)={state[1]:7.3f}  θ_dot={state[2]:7.3f}  done={done}")

print()
print("  这 5 步的总奖励 total_reward =", total_reward)
print()

# -----------------------------------------------------------------------------
# 5. 收尾
# -----------------------------------------------------------------------------
env.close()

print("=" * 50)
print("第一步完成！")
print("  - 你知道了：状态(obs)、动作(action)、奖励(reward)、done、info")
print("  - 你知道了：reset -> step -> step -> ... -> done 的基本循环")
print()
print("  下一步我们会写「环境封装」，把 reset/step 包成好用的接口。")
print("  理解透第一步后，告诉我，我们继续第二步。")
print("=" * 50)
