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
# 1. 计算 returns（折扣奖励）和 GAE（广义优势估计）
# =============================================================================
# 从后往前：G_t = r_t + gamma * G_{t+1}，若 done 则 G_{t+1} 不继续累加。
# 
# 【学习点1：为什么需要GAE？】
# - 简单优势：A_t = R_t - V(s_t)，方差高（因为R_t是蒙特卡洛估计）
# - GAE：结合TD误差，平衡偏差和方差
# - GAE(λ) = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
#   其中 δ_t = r_t + γV(s_{t+1}) - V(s_t) 是TD误差
# - λ=0: 纯TD（低方差，高偏差）
# - λ=1: 纯蒙特卡洛（高方差，低偏差）
# - λ=0.95: 常用折中值
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


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """
    计算广义优势估计（GAE）- 单轨迹版本
    
    【学习点2：GAE的数学原理】
    - TD误差：δ_t = r_t + γV(s_{t+1}) - V(s_t)
    - GAE：A_t^GAE = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
    
    Args:
        rewards: (T,) array，奖励序列
        values: (T,) array，当前状态的值函数 V(s_t)
        next_values: (T,) array，下一状态的值函数 V(s_{t+1})
        dones: (T,) bool array，是否结束
        gamma: 折扣因子
        gae_lambda: GAE参数，控制偏差-方差权衡
    
    Returns:
        advantages: (T,) array，优势估计
        returns: (T,) array，用于训练Critic的returns
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    
    # 从后往前计算GAE
    for t in reversed(range(T)):
        # 如果done，下一状态的值函数为0，gae也不传递
        mask = 1.0 - float(dones[t])
        
        # TD误差：δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        
        # GAE累加：gae = δ_t + (γλ) * gae_{t+1} * (1-done)
        # 关键：如果done=True，gae不从下一步传递
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    
    # returns = advantages + values（用于训练Critic）
    returns = advantages + values
    
    return advantages, returns


def compute_gae_vectorized(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95, num_envs=1):
    """
    【关键修复】为向量化环境正确计算GAE
    
    问题：当使用并行环境时，buffer中的数据是按step存储的：
        step0: [env0, env1, ..., envN]
        step1: [env0, env1, ..., envN]
        ...
    
    但GAE需要按每个环境的轨迹计算：
        env0: step0 -> step1 -> step2 -> ...
        env1: step0 -> step1 -> step2 -> ...
    
    Args:
        rewards: (num_envs * num_steps,) array
        values: (num_envs * num_steps,) array
        next_values: (num_envs * num_steps,) array
        dones: (num_envs * num_steps,) bool array
        num_envs: 并行环境数量
    
    Returns:
        advantages: (num_envs * num_steps,) array
        returns: (num_envs * num_steps,) array
    """
    total_samples = len(rewards)
    num_steps = total_samples // num_envs
    
    if total_samples % num_envs != 0:
        print(f"警告：样本数({total_samples})不能被环境数({num_envs})整除，回退到简单GAE")
        return compute_gae(rewards, values, next_values, dones, gamma, gae_lambda)
    
    # 重塑为 (num_steps, num_envs) - 按step分组
    rewards = rewards.reshape(num_steps, num_envs)
    values = values.reshape(num_steps, num_envs)
    next_values = next_values.reshape(num_steps, num_envs)
    dones = dones.reshape(num_steps, num_envs)
    
    advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
    gae = np.zeros(num_envs, dtype=np.float32)  # 每个环境独立的gae
    
    # 从后往前计算GAE
    for t in reversed(range(num_steps)):
        # 掩码：如果done=True，gae不传递
        mask = 1.0 - dones[t].astype(np.float32)
        
        # TD误差
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        
        # GAE累加（每个环境独立）
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    
    # returns = advantages + values
    returns = advantages + values
    
    # 重塑回 (num_envs * num_steps,)
    return advantages.flatten(), returns.flatten()


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
               gamma=0.99, eps_clip=0.2, k_epochs=5, gae_lambda=0.95, use_gae=True,
               value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, num_envs=1):
    """
    用 buffer 里的经验做一次 PPO 更新。
    
    【学习点3：PPO更新的完整流程】
    1. 计算优势（GAE或简单方法）
    2. 归一化优势（减少方差）
    3. 多轮更新（k_epochs）：
       - 计算新策略的log_prob
       - 计算ratio = π_new / π_old
       - PPO裁剪目标：min(ratio*A, clip(ratio)*A)
       - 可选：添加熵奖励（鼓励探索）
       - Critic损失：MSE(V(s), returns)
    
    Args:
        use_gae: 是否使用GAE（推荐True）
        value_coef: Critic损失的权重
        entropy_coef: 熵奖励的权重（鼓励探索）
        max_grad_norm: 梯度裁剪阈值
        num_envs: 并行环境数量（用于正确计算GAE）
    """
    if buffer.size() == 0:
        return None, None, None, None, None

    states = torch.from_numpy(np.array(buffer.states, dtype=np.float32)).to(device)
    actions = torch.from_numpy(np.array(buffer.actions, dtype=np.float32)).to(device)
    rewards = np.array(buffer.rewards, dtype=np.float32)
    dones = np.array(buffer.dones, dtype=bool)
    old_log_probs = torch.from_numpy(np.array(buffer.log_probs, dtype=np.float32)).to(device)

    # 计算值函数（用于GAE或简单优势）
    with torch.no_grad():
        values = critic(states).squeeze(-1).cpu().numpy()
    
    # 计算优势（GAE或简单方法）
    if use_gae and hasattr(buffer, 'next_states') and len(buffer.next_states) > 0:
        # 【优化1：使用GAE】需要next_states和next_values
        # 【学习点：GAE的正确计算】
        # - 对于每个(s_t, a_t, r_t, s_{t+1})，我们需要V(s_{t+1})
        # - 如果done[t]=True，说明episode结束，V(s_{t+1})=0
        # - 否则，V(s_{t+1}) = critic(next_states[t])
        next_states = torch.from_numpy(np.array(buffer.next_states, dtype=np.float32)).to(device)
        with torch.no_grad():
            next_values_all = critic(next_states).squeeze(-1).cpu().numpy()
        
        # 【关键修复】检查长度是否匹配
        if len(next_values_all) != len(rewards):
            print(f"警告：next_values_all长度({len(next_values_all)})与rewards长度({len(rewards)})不匹配，回退到简单方法")
            # 回退到简单方法
            returns = compute_returns(rewards, dones, gamma)
            returns = torch.from_numpy(np.array(returns, dtype=np.float32)).to(device)
            advantages = returns.cpu().numpy() - values
            advantages = torch.from_numpy(advantages.astype(np.float32)).to(device)
        else:
            # 【关键修复】使用向量化GAE计算，正确处理并行环境
            # 当 num_envs > 1 时，使用 compute_gae_vectorized
            if num_envs > 1:
                advantages, returns = compute_gae_vectorized(
                    rewards, values, next_values_all, dones, gamma, gae_lambda, num_envs
                )
            else:
                # 单环境：构建next_values
                next_values = np.zeros_like(values)
                for i in range(len(rewards)):
                    if not dones[i]:
                        next_values[i] = next_values_all[i]
                advantages, returns = compute_gae(
                    rewards, values, next_values, dones, gamma, gae_lambda
                )
            
            # 【修复】将advantages和returns都转换为tensor
            advantages = torch.from_numpy(advantages.astype(np.float32)).to(device)
            returns = torch.from_numpy(returns.astype(np.float32)).to(device)
    else:
        # 简单方法：advantages = returns - V(s)
        returns = compute_returns(rewards, dones, gamma)
        returns = torch.from_numpy(np.array(returns, dtype=np.float32)).to(device)
        advantages = returns.cpu().numpy() - values
        advantages = torch.from_numpy(advantages.astype(np.float32)).to(device)
    
    # 【优化2：归一化优势】减少方差，提升稳定性
    # 【稳定性】检查advantages是否异常
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std < 1e-8:
        # 如果标准差太小，说明advantages几乎相同，只做中心化
        advantages = advantages - adv_mean
    else:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    
    # 【稳定性】检查归一化后是否有异常值
    if torch.any(torch.isnan(advantages)) or torch.any(torch.isinf(advantages)):
        print("警告：advantages归一化后出现NaN/Inf，重新计算advantages")
        # 如果归一化后还有NaN，说明原始数据有问题，回退到简单方法
        with torch.no_grad():
            values_simple = critic(states).squeeze(-1)
        advantages = returns - values_simple
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 如果还是NaN，说明数据严重异常，设为0
        if torch.any(torch.isnan(advantages)) or torch.any(torch.isinf(advantages)):
            print("严重警告：advantages仍然异常，设为0")
            advantages = torch.zeros_like(advantages)
    
    # 【优化3：归一化returns】帮助Critic训练更稳定
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    returns_unnorm = returns  # 保存原始returns用于计算loss

    actor_losses = []
    critic_losses = []
    entropies = []
    kl_divs = []

    # =====================================================================
    # 【核心概念2：Loss降低意味着什么？】
    # =====================================================================
    # 
    # 【Actor Loss降低意味着什么？】
    # - Actor Loss = -min(ratio*A, clip(ratio)*A) - entropy_coef*entropy
    # - Loss降低 → 策略在"好动作"上的概率增加
    # - 但注意：这是相对于"优势A"来说的
    #   * 如果A>0（好动作），loss降低 → 增加这个动作的概率 ✓
    #   * 如果A<0（坏动作），loss降低 → 减少这个动作的概率 ✓
    # - 所以：Actor Loss降低 = 策略在朝着"高奖励方向"改进
    #
    # 【Critic Loss降低意味着什么？】
    # - Critic Loss = MSE(V(s), returns)
    # - Loss降低 → 值函数V(s)预测更准确
    # - 值函数准确 → 优势A = returns - V(s) 更准确
    # - 优势准确 → Actor更新更准确 → 训练更稳定
    #
    # 【重要：Loss不是越低越好！】
    # - 如果Actor Loss降得太快 → 可能过拟合到当前经验
    # - 如果Critic Loss降得太快 → 可能值函数过拟合
    # - 理想情况：Loss平稳下降，reward稳步上升
    #
    # =====================================================================

    for _ in range(k_epochs):
        # 新策略下 (s, a) 的 log_prob
        new_log_probs = compute_log_prob(actor, states, actions)
        
        # =====================================================================
        # 【核心概念1：熵（Entropy）vs KL散度（KL Divergence）】
        # =====================================================================
        # 
        # 【熵 H(π) = -Σ π(a|s) log π(a|s)】
        # - 衡量"策略的随机性/不确定性"
        # - 高熵：策略很随机，所有动作概率相近 → 探索多
        # - 低熵：策略很确定，某个动作概率很高 → 利用多
        # - 例子：
        #   * 高熵：π(左)=0.3, π(右)=0.3, π(中)=0.4 → 探索各种动作
        #   * 低熵：π(左)=0.01, π(右)=0.01, π(中)=0.98 → 几乎总是选中间
        #
        # 【KL散度 KL(π_old || π_new) = Σ π_old(a|s) log(π_old/π_new)】
        # - 衡量"新旧策略的差异程度"
        # - 高KL：策略变化大（可能破坏之前学到的知识）
        # - 低KL：策略变化小（稳定更新）
        # - 注意：KL散度需要两个分布，熵只需要一个分布
        #
        # 【为什么熵鼓励探索？】
        # - 在loss中：actor_loss = actor_loss - entropy_coef * entropy
        # - 因为我们要最小化loss，所以减去熵 = 最大化熵
        # - 最大化熵 → 策略更随机 → 尝试更多动作 → 探索更多
        # - 这是"探索-利用权衡"中的探索部分
        #
        # =====================================================================
        mean, log_std = actor(states)
        std = torch.exp(log_std)
        normal = dist.Normal(mean, std)
        entropy = normal.entropy().sum(dim=-1).mean()  # 计算熵

        # ratio = π_new(a|s) / π_old(a|s)
        # 【稳定性】限制log_ratio范围，防止exp溢出
        log_ratio = new_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)  # 防止exp溢出
        ratio = torch.exp(log_ratio)

        # 【学习点5：PPO裁剪】防止策略更新过大
        # surr1 = ratio * A（未裁剪）
        # surr2 = clip(ratio) * A（裁剪后）
        # 取min确保不会过度优化
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 【优化4：添加熵奖励】鼓励探索
        # actor_loss降低 = 策略变好 所以模型会倾向于像高熵方向走

        actor_loss = actor_loss - entropy_coef * entropy

        # 【学习点6：Critic损失】学习值函数V(s)
        values_pred = critic(states).squeeze(-1)
        
        # 【稳定性】检查returns和values_pred是否异常
        if torch.any(torch.isnan(returns_unnorm)) or torch.any(torch.isinf(returns_unnorm)):
            print("警告：returns包含NaN/Inf，跳过Critic更新")
            critic_loss = torch.tensor(0.0, device=device)
        else:
            # 【优化】使用 Huber Loss (smooth_l1_loss) 替代 MSE
            # Huber Loss 对异常值更鲁棒，可以防止 Critic Loss 过大导致的训练不稳定
            critic_loss = F.smooth_l1_loss(values_pred, returns_unnorm) * value_coef 
        # actor loss一般比较低 因为advantage被norm过 而且这里采用的mse方法，
        # 所以乘以一个系数，避免value网络loss过高，模型专注于降低value的loss，而忽视了actor的loss，就是避免和actor的更新抢梯度

        # 【学习点7：KL散度】监控策略变化
        # KL(π_old || π_new) = E[log π_old - log π_new]
        # 注意：这个值可能为负（如果新策略概率更高），但KL散度应该是非负的
        # 我们使用绝对值或更准确的近似：KL ≈ |mean(old_log_prob - new_log_prob)|
        # 或者使用更准确的公式：KL ≈ mean(exp(new_log_prob - old_log_prob) * (new_log_prob - old_log_prob))
        # 【稳定性】使用已经clamp过的log_ratio（上面已经计算过）
        kl_div = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()  # 更准确的KL散度近似
        # 如果KL太大（>0.01），说明策略变化太快，可能需要减小学习率或提前停止
        # 【稳定性】检查KL是否异常
        if np.isnan(kl_div) or np.isinf(kl_div):
            kl_div = 0.0  # 如果异常，设为0
        
        # 【优化5：KL散度早停】如果策略变化太大，提前停止更新
        if kl_div > 0.02:  # 阈值可调
            # 策略变化太大，停止本轮更新，防止破坏已学到的知识
            break

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        
        #计算梯度时，可能梯度会非常大，所以需要裁剪梯度，避免梯度爆炸
        # 【稳定性】检查梯度是否正常
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        
        # 如果梯度异常，跳过更新
        if torch.isnan(actor_grad_norm) or torch.isnan(critic_grad_norm) or \
           torch.isinf(actor_grad_norm) or torch.isinf(critic_grad_norm):
            print(f"警告：梯度异常，跳过本次更新。Actor Grad: {actor_grad_norm}, Critic Grad: {critic_grad_norm}")
            break
        
        opt_actor.step()
        opt_critic.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropies.append(entropy.item())
        kl_divs.append(kl_div)

    # 处理advantages：可能是tensor或numpy数组
    if isinstance(advantages, torch.Tensor):
        mean_advantage = np.mean(advantages.cpu().numpy())
    else:
        mean_advantage = np.mean(advantages)
    
    return (np.mean(actor_losses), np.mean(critic_losses), 
            np.mean(entropies), np.mean(kl_divs), mean_advantage)


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
        al, cl, entropy, kl_div, mean_adv = ppo_update(env, actor, critic, buffer, opt_actor, opt_critic, device,
                            gamma=gamma, eps_clip=eps_clip, k_epochs=k_epochs)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1:3d}/{max_episodes}  reward={total_reward:8.2f}  steps={length:3d}  "
                  f"actor_loss={al:.4f}  critic_loss={cl:.4f}  entropy={entropy:.4f}  kl={kl_div:.4f}")

    env.close()

    print("\n" + "=" * 50)
    print("第七步完成！")
    print("  - PPO 裁剪目标 + Critic MSE 已实现")
    print("  - 短训练跑完，若 reward 有升、loss 在动，说明更新正常")
    print("  - 下一步：完整训练循环 + 画 reward 曲线 + 保存模型。")
    print("=" * 50)
