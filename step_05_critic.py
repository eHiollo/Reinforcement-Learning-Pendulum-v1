"""
第五步：实现价值网络 (Critic)

再写一个 MLP：state -> value（一个实数）。
不训练，只做前向传播，体会「Critic 评价当前状态」的用法。

运行：python step_05_critic.py
"""

import numpy as np
import torch
import torch.nn as nn

from step_02_env_wrapper import PendulumWrapper
from step_04_actor import Actor, actor_select_action


# =============================================================================
# 1. Critic 网络：state -> value
# =============================================================================
# 输入 state (state_dim,)，输出一个实数，表示「从这个状态出发的期望回报」。
# 结构类似 Actor，但最后一层输出 1 维，不做 tanh。
# =============================================================================


class Critic(nn.Module):
    """
    价值网络：给定 state，输出一个标量 value。
    结构：state -> Linear -> ReLU -> Linear -> ReLU -> Linear -> value
    """

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim) 或 (state_dim,)
        返回: (batch, 1) 或 (1,)，标量
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# =============================================================================
# 2. 把 state (numpy) 送给 Critic，拿到 value (float)
# =============================================================================
def critic_value(critic: nn.Module, state: np.ndarray, device: torch.device) -> float:
    """
    critic: 价值网络
    state: (state_dim,) numpy
    返回: 标量 float
    """
    x = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        v = critic(x)
    return v.squeeze().cpu().item()


# =============================================================================
# 3. Actor + Critic 一起跑几步（都不训练）
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("第五步：价值网络 (Critic)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PendulumWrapper("Pendulum-v1")

    actor = Actor(env.state_dim, env.action_dim, hidden=64).to(device)
    critic = Critic(env.state_dim, hidden=64).to(device)

    print(f"\nCritic 结构: state_dim={env.state_dim} -> 64 -> 64 -> 1")
    print("当前未训练，value 仅作示意。\n")

    state = env.reset(seed=42)
    total_reward = 0.0

    print("用 Actor 选动作、Critic 评 state，跑 10 步：")
    for t in range(10):
        action = actor_select_action(actor, state, device)
        v = critic_value(critic, state, device)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward

        print(f"  t={t+1:2d}  state[0:2]=({state[0]:.3f},{state[1]:.3f})  "
              f"action={action[0]:.3f}  reward={reward:.3f}  value(s)={v:.3f}  done={done}")
        state = next_state
        if done:
            break

    print(f"\n10 步总奖励: {total_reward:.3f}")
    env.close()

    print("\n" + "=" * 50)
    print("第五步完成！")
    print("  - Critic: state -> value，已用 MLP 实现")
    print("  - 下一步：收集经验 + 理解 PPO，再实现更新。")
    print("=" * 50)
