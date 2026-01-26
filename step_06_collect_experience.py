"""
第六步：收集经验并理解 PPO

用当前 Actor 在环境里跑，存 (s, a, r, s', done, log_prob)。
理解「为什么要用 PPO」和「on-policy vs off-policy」的直观区别。
先不实现梯度更新，只存经验。

运行：python step_06_collect_experience.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

from step_02_env_wrapper import PendulumWrapper


# =============================================================================
# 1. PPO 用的 Actor：输出 mean 和 log_std，能从分布采样并算 log_prob
# =============================================================================
# 之前的 Actor 直接输出 action，PPO 需要知道「选这个 action 的概率」。
# 所以我们需要一个能输出分布参数的 Actor。
# =============================================================================


class PPOActor(nn.Module):
    """
    PPO 用的策略网络：输出 mean 和 log_std，能从正态分布采样。
    结构：state -> Linear -> ReLU -> Linear -> ReLU -> (mean, log_std)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_layer = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        """
        返回: (mean, log_std)
        mean: (batch, action_dim)
        log_std: (batch, action_dim)
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)  # 限制范围
        return mean, log_std


# =============================================================================
# 2. 经验缓冲区：存 (s, a, r, s', done, log_prob)
# =============================================================================
# 强化学习需要「经验」来更新策略。我们把每一步的 transition 存起来。
# =============================================================================


class ExperienceBuffer:
    """
    经验缓冲区：存一个 episode 或一段轨迹的 (s, a, r, done, log_prob)。
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def add(self, state, action, reward, done, log_prob):
        """存一个 transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def size(self):
        """返回经验数量"""
        return len(self.states)


# =============================================================================
# 3. 用 Actor 选动作，并计算 log_prob（PPO 需要）
# =============================================================================
# PPO 是 on-policy 算法，需要知道「选这个动作的概率」。
# 我们用正态分布采样，并记录 log_prob。
# =============================================================================


def actor_select_action_with_log_prob(actor: nn.Module, state: np.ndarray, device: torch.device):
    """
    用 PPOActor 选动作，并返回 log_prob（PPO 更新时需要）。

    返回: (action_numpy, log_prob_float)
    """
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

    with torch.no_grad():
        mean, log_std = actor(state_tensor)
        std = torch.exp(log_std)

        # 从正态分布采样
        normal = dist.Normal(mean, std)
        action_unbounded = normal.sample()
        log_prob = normal.log_prob(action_unbounded).sum(dim=-1)

        # 限制到 [-2, 2]（Pendulum 的动作范围）
        action = torch.tanh(action_unbounded) * 2.0

        # 调整 log_prob（因为用了 tanh 变换）
        # tanh 的雅可比：1 - tanh^2(x)
        log_prob -= torch.log(1 - torch.tanh(action_unbounded).pow(2) + 1e-6).sum(dim=-1)
        log_prob -= np.log(2.0)  # 乘以 2 的雅可比

    return action.squeeze(0).cpu().numpy(), log_prob.item()


# =============================================================================
# 4. 收集经验：用 Actor 在环境里跑，存到缓冲区
# =============================================================================


def collect_experience(env, actor, buffer, device, max_steps=200, seed=None):
    """
    用当前 Actor 在环境里跑一个 episode，把经验存到 buffer。
    """
    state = env.reset(seed=seed)
    total_reward = 0.0

    for step in range(max_steps):
        action, log_prob = actor_select_action_with_log_prob(actor, state, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存经验
        buffer.add(state, action, reward, done, log_prob)

        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward, step + 1


# =============================================================================
# 4. 理解 PPO：为什么需要、on-policy vs off-policy
# =============================================================================
# 这部分用注释和打印说明，不写代码。
# =============================================================================


if __name__ == "__main__":
    print("=" * 50)
    print("第六步：收集经验并理解 PPO")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PendulumWrapper("Pendulum-v1")
    actor = PPOActor(env.state_dim, env.action_dim, hidden=64).to(device)
    buffer = ExperienceBuffer()

    print("\n【理解 PPO】")
    print("=" * 50)
    print("1. 为什么要用 PPO？")
    print("   - 策略梯度算法（如 REINFORCE）每次更新都要用「当前策略」收集的新数据")
    print("   - 如果策略更新太大，旧数据就不能用了（on-policy 限制）")
    print("   - PPO 通过「裁剪」限制更新幅度，让旧数据能多用几次，提高样本效率")
    print()
    print("2. On-policy vs Off-policy")
    print("   - On-policy（如 PPO）：只能用「当前策略」收集的数据更新")
    print("   - Off-policy（如 DQN）：可以用「旧策略」收集的数据更新")
    print("   - PPO 是 on-policy，但通过裁剪让数据能复用几次（伪 off-policy）")
    print("=" * 50)

    print("\n【收集经验】用当前 Actor 跑 3 个 episode，存到 buffer：\n")

    for ep in range(3):
        buffer.clear()
        reward, length = collect_experience(env, actor, buffer, device, max_steps=200, seed=42 + ep)

        print(f"Episode {ep + 1}:")
        print(f"  总奖励: {reward:.2f}")
        print(f"  步数: {length}")
        print(f"  收集的经验数: {buffer.size()}")
        print(f"  前 3 个 transition:")
        for i in range(min(3, buffer.size())):
            s = buffer.states[i]
            a = buffer.actions[i]
            r = buffer.rewards[i]
            lp = buffer.log_probs[i]
            print(f"    [{i}] state=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})  "
                  f"action={a[0]:.3f}  reward={r:.3f}  log_prob={lp:.3f}")
        print()

    env.close()

    print("=" * 50)
    print("第六步完成！")
    print("  - 已实现经验缓冲区，能存 (s, a, r, done, log_prob)")
    print("  - 理解了 PPO 的动机：限制更新幅度，提高样本效率")
    print("  - 下一步：实现 PPO 的更新逻辑（目标函数 + 裁剪）。")
    print("=" * 50)
