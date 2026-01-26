"""
查看 Reward 的计算

演示：
1. Reward 是环境自动计算的
2. Reward 和状态、动作的关系
3. 如何自定义 reward（可选）

运行：python check_reward.py
"""

import numpy as np
import gymnasium as gym

# 创建环境
env = gym.make("Pendulum-v1")

print("=" * 50)
print("Reward 计算说明")
print("=" * 50)

print("\n1. Reward 是环境自动计算的")
print("   我们调用 env.step(action)，环境内部会：")
print("   - 用物理公式计算下一个状态")
print("   - 根据状态和动作计算 reward")
print("   - 返回给我们\n")

# 跑几步，观察 reward
print("2. 观察 Reward 和状态的关系：\n")
state, _ = env.reset(seed=42)

for step in range(5):
    action = env.action_space.sample()  # 随机动作
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    # 从 state 反推角度（近似）
    # state = [cos(θ), sin(θ), θ_dot]
    cos_theta = next_state[0]
    sin_theta = next_state[1]
    theta_dot = next_state[2]
    
    # 角度（弧度）
    theta = np.arctan2(sin_theta, cos_theta)
    
    print(f"Step {step + 1}:")
    print(f"  角度 θ ≈ {theta:.4f} rad ({np.degrees(theta):.2f}°)")
    print(f"  角速度 θ_dot = {theta_dot:.4f}")
    print(f"  动作 u = {action[0]:.4f}")
    print(f"  Reward = {reward:.4f}")
    print(f"  （Reward 公式：-(θ² + 0.1·θ_dot² + 0.001·u²)）")
    print()
    
    state = next_state

env.close()

print("3. Reward 的特点：")
print("   - Reward 是负数（表示成本）")
print("   - 越接近 0 越好（摆竖直向上、静止、不施力时 reward ≈ 0）")
print("   - 角度越大、角速度越大、动作越大，reward 越小（越负）")
print()

print("=" * 50)
print("总结：Reward 是环境自动计算的，我们只需要用 env.step() 获取。")
print("=" * 50)
