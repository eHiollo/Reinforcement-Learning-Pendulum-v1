"""
第四步：实现简单策略网络 (Actor)

用 PyTorch 写一个小 MLP：state -> action。
不训练，只用随机初始化权重，在环境里跑几步，体会「神经网络当策略」的感觉。

运行：python step_04_actor.py
"""

import numpy as np
import torch
import torch.nn as nn

from step_02_env_wrapper import PendulumWrapper


# =============================================================================
# 1. Actor 网络：state -> action
# =============================================================================
# 输入 state 形状 (state_dim,)，输出 action 形状 (action_dim,)，范围 [-2, 2]。
# 用两层隐藏层 + 最后一层 tanh * 2 把输出压到 [-2, 2]。
# =============================================================================


class Actor(nn.Module):
    """
    策略网络：给定 state，输出 action。
    结构：state -> Linear -> ReLU -> Linear -> ReLU -> Linear -> tanh * 2
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim) 或 (state_dim,)
        返回: (batch, action_dim) 或 (action_dim,)，范围 [-2, 2]
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # 压到 [-1, 1] 再乘 2 -> [-2, 2]，符合 Pendulum 动作范围
        action = torch.tanh(x) * 2.0
        return action


# =============================================================================
# 2. 把 state (numpy) 送给 Actor，拿到 action (numpy)
# =============================================================================
def actor_select_action(actor: nn.Module, state: np.ndarray, device: torch.device) -> np.ndarray:
    """
    actor: 策略网络
    state: (state_dim,) numpy
    返回: (action_dim,) numpy，范围 [-2, 2]
    """
    x = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action = actor(x)
    return action.squeeze(0).cpu().numpy()


# =============================================================================
# 3. 用 Actor 在环境里跑几步（不训练）
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("第四步：简单策略网络 (Actor)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PendulumWrapper("Pendulum-v1")

    # 建 Actor，随机初始化，不训练
    actor = Actor(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden=64,
    ).to(device)

    print(f"\nActor 结构: state_dim={env.state_dim} -> 64 -> 64 -> action_dim={env.action_dim}")
    print("当前未训练，相当于随机策略。\n")

    state = env.reset(seed=42)
    total_reward = 0.0

    print("用 Actor(state) 选动作，跑 10 步：")
    for t in range(10):
        action = actor_select_action(actor, state, device)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward

        print(f"  t={t+1:2d}  state[0:2]=({state[0]:.3f},{state[1]:.3f})  "
              f"action={action[0]:.3f}  reward={reward:.3f}  done={done}")
        state = next_state
        if done:
            break

    print(f"\n10 步总奖励: {total_reward:.3f}")
    env.close()

    print("\n" + "=" * 50)
    print("第四步完成！")
    print("  - Actor: state -> action，已用 MLP 实现")
    print("  - 下一步：实现价值网络 (Critic)，state -> value。")
    print("=" * 50)
