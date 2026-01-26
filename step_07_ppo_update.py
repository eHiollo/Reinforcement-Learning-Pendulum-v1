"""
第七步：实现 PPO 更新

实现「目标函数 + 裁剪」、Critic 的 MSE 损失，
把 Actor、Critic 的更新串起来。
跑一个很短的训练（几十个 episode），验证 loss 会动。

运行：python step_07_ppo_update.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

from step_02_env_wrapper import PendulumWrapper
from step_05_critic import Critic
from step_06_collect_experience import (
    PPOActor,
    ExperienceBuffer,
    actor_select_action_with_log_prob,
    collect_experience,
)


# =============================================================================
# 1. 计算 returns（折扣奖励）
# =============================================================================
# 从后往前：G_t = r_t + gamma * G_{t+1}，若 done 则 G_{t+1} 不继续累加。
# =============================================================================


def compute_returns(rewards, dones, gamma=0.99):
    """
    rewards: list of float
    dones: list of bool
    返回: list of float，与 rewards 等长
    """
    returns = []
    g = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            g = 0.0
        g = r + gamma * g
        returns.append(g)
    return list(reversed(returns))


# =============================================================================
# 2. 给定 (s, a)，用当前 Actor 算 log_prob
# =============================================================================
# 更新时要用「新策略」对旧动作 a 的 log 概率，用于算 ratio。
# a 是 [-2, 2] 的，先反解出 tanh 前的 x，再算 log_prob。
# =============================================================================


def compute_log_prob(actor, states, actions):
    """
    states: (N, state_dim) tensor
    actions: (N, action_dim) tensor，范围 [-2, 2]
    返回: (N,) tensor，每个 (s,a) 在当前策略下的 log_prob
    """
    mean, log_std = actor(states)
    std = torch.exp(log_std)

    # a = tanh(x) * 2  =>  x = atanh(a / 2)
    a_norm = actions / 2.0
    a_norm = torch.clamp(a_norm, -0.999, 0.999)
    x = torch.atanh(a_norm)

    normal = dist.Normal(mean, std)
    log_prob = normal.log_prob(x).sum(dim=-1)
    log_prob -= torch.log(1 - a_norm.pow(2) + 1e-6).sum(dim=-1)
    log_prob -= np.log(2.0)
    return log_prob


# =============================================================================
# 3. PPO 更新：Actor 裁剪目标 + Critic MSE
# =============================================================================


def ppo_update(env, actor, critic, buffer, opt_actor, opt_critic, device,
               gamma=0.99, eps_clip=0.2, k_epochs=5):
    """
    用 buffer 里的经验做一次 PPO 更新。
    """
    if buffer.size() == 0:
        return None, None

    states = torch.from_numpy(np.array(buffer.states, dtype=np.float32)).to(device)
    actions = torch.from_numpy(np.array(buffer.actions, dtype=np.float32)).to(device)
    rewards = np.array(buffer.rewards, dtype=np.float32)
    dones = np.array(buffer.dones, dtype=bool)
    old_log_probs = torch.from_numpy(np.array(buffer.log_probs, dtype=np.float32)).to(device)

    # returns
    returns = compute_returns(rewards, dones, gamma)
    returns = torch.from_numpy(np.array(returns, dtype=np.float32)).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # advantages = returns - V(s)
    with torch.no_grad():
        values = critic(states).squeeze(-1)
        advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_losses = []
    critic_losses = []

    for _ in range(k_epochs):
        # 新策略下 (s, a) 的 log_prob
        new_log_probs = compute_log_prob(actor, states, actions)

        # ratio = π(a|s) / π_old(a|s)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic：MSE(V(s), returns)
        values = critic(states).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        opt_actor.step()
        opt_critic.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    return np.mean(actor_losses), np.mean(critic_losses)


# =============================================================================
# 4. 训练循环：收集 -> 更新 -> 重复
# =============================================================================


if __name__ == "__main__":
    print("=" * 50)
    print("第七步：PPO 更新 + 短训练")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PendulumWrapper("Pendulum-v1")

    actor = PPOActor(env.state_dim, env.action_dim, hidden=64).to(device)
    critic = Critic(env.state_dim, hidden=64).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)

    buffer = ExperienceBuffer()

    max_episodes = 80
    max_steps = 200
    gamma = 0.99
    eps_clip = 0.2
    k_epochs = 5

    print(f"\n配置: max_episodes={max_episodes}, max_steps={max_steps}, gamma={gamma}, eps_clip={eps_clip}, k_epochs={k_epochs}\n")

    for ep in range(max_episodes):
        buffer.clear()
        total_reward, length = collect_experience(env, actor, buffer, device, max_steps=max_steps, seed=42 + ep)
        al, cl = ppo_update(env, actor, critic, buffer, opt_actor, opt_critic, device,
                            gamma=gamma, eps_clip=eps_clip, k_epochs=k_epochs)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1:3d}/{max_episodes}  reward={total_reward:8.2f}  steps={length:3d}  "
                  f"actor_loss={al:.4f}  critic_loss={cl:.4f}")

    env.close()

    print("\n" + "=" * 50)
    print("第七步完成！")
    print("  - PPO 裁剪目标 + Critic MSE 已实现")
    print("  - 短训练跑完，若 reward 有升、loss 在动，说明更新正常")
    print("  - 下一步：完整训练循环 + 画 reward 曲线 + 保存模型。")
    print("=" * 50)
