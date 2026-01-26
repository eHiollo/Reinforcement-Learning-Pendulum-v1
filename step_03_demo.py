"""
第三步：策略 vs 价值 —— 用「假」函数体会接口

不写神经网络，只写两个简单的 stub：
  - policy(state) -> action   （策略：怎么动）
  - value(state) -> float    （价值：有多好）

跑几步，看看它们怎么插进 reset -> step 循环里。

运行：python step_03_demo.py
"""

import numpy as np
from step_02_env_wrapper import PendulumWrapper


# -----------------------------------------------------------------------------
# 1. 「假」策略：输入 state，输出 action
# -----------------------------------------------------------------------------
def policy(state):
    """
    策略： state -> action
    这里随便用随机动作代替，后面会换成神经网络。
    """
    # 动作范围 [-2, 2]，这里随机一个
    action = np.random.uniform(-2.0, 2.0, size=1).astype(np.float32)
    return action


# -----------------------------------------------------------------------------
# 2. 「假」价值：输入 state，输出一个实数
# -----------------------------------------------------------------------------
def value(state):
    """
    价值： state -> 一个数
    这里用 |cos| + |sin| 瞎凑一个「好坏」示意，后面会换成神经网络。
    """
    # 越接近竖直 cos≈1, sin≈0，这里随便用个正相关
    return float(np.abs(state[0]) + np.abs(state[1]))


# -----------------------------------------------------------------------------
# 3. 用 Wrapper + 假策略 / 假价值 跑几步
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("第三步：策略 vs 价值 —— 接口演示")
    print("=" * 50)

    env = PendulumWrapper("Pendulum-v1")
    np.random.seed(42)

    state = env.reset(seed=42)
    print(f"\n初始 state = {state}")
    print(f"value(state) = {value(state):.4f}  （假价值，仅示意）\n")

    print("用 policy(state) 选动作，执行 5 步：")
    for t in range(5):
        action = policy(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        v = value(state)
        print(f"  t={t+1}  state[0:2]={state[0]:.3f},{state[1]:.3f}  "
              f"action={action[0]:.3f}  reward={reward:.3f}  "
              f"value(s)={v:.3f}  done={done}")
        state = next_state

    env.close()

    print("\n" + "=" * 50)
    print("第三步完成！")
    print("  - policy(s) -> a  负责「怎么做」")
    print("  - value(s) -> 数  负责「有多好」")
    print("  下一步：用 PyTorch 实现真正的 policy 网络 (Actor)。")
    print("=" * 50)
