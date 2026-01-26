"""
第八步：完整训练循环

整合前面所有步骤，写一个完整的训练脚本：
- 循环「收集经验 → PPO 更新 → 记录 reward」
- 画 reward 曲线（移动平均）
- 保存模型
- 跑足够长时间，看到倒立摆能立住

运行：python step_08_train.py
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os

from step_02_env_wrapper import PendulumWrapper
from step_05_critic import Critic
from step_06_collect_experience import (
    PPOActor,
    ExperienceBuffer,
    collect_experience,
)
from step_07_ppo_update import ppo_update


# =============================================================================
# 1. 画训练曲线
# =============================================================================


def plot_training_curve(episode_rewards, save_path="./training_curve.png", window=50):
    """
    画 reward 曲线：原始值 + 移动平均
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # 原始 reward（半透明）
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # 移动平均
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, 
                label=f'Moving Average ({window} episodes)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward Curve')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


# =============================================================================
# 2. 保存模型
# =============================================================================


def save_model(actor, critic, opt_actor, opt_critic, filepath):
    """保存 Actor 和 Critic 的模型和优化器状态"""
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': opt_actor.state_dict(),
        'critic_optimizer_state_dict': opt_critic.state_dict(),
    }, filepath)
    print(f"模型已保存到: {filepath}")


# =============================================================================
# 3. 完整训练循环
# =============================================================================


def train(
    max_episodes=500,
    max_steps_per_episode=200,
    gamma=0.99,
    eps_clip=0.2,
    k_epochs=5,
    lr_actor=3e-4,
    lr_critic=3e-4,
    hidden_dim=64,
    save_frequency=100,
    model_dir="./models",
    seed=42,
):
    """
    完整训练循环
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 环境
    env = PendulumWrapper("Pendulum-v1")
    
    # 网络和优化器
    actor = PPOActor(env.state_dim, env.action_dim, hidden=hidden_dim).to(device)
    critic = Critic(env.state_dim, hidden=hidden_dim).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    
    buffer = ExperienceBuffer()
    
    # 记录训练过程
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)  # 最近 100 个 episode 的平均
    
    print("=" * 50)
    print("开始训练倒立摆强化学习智能体")
    print("=" * 50)
    print(f"\n配置:")
    print(f"  - 最大 episode 数: {max_episodes}")
    print(f"  - 每 episode 最大步数: {max_steps_per_episode}")
    print(f"  - 折扣因子 gamma: {gamma}")
    print(f"  - PPO 裁剪参数 eps_clip: {eps_clip}")
    print(f"  - 每次更新轮数 k_epochs: {k_epochs}")
    print(f"  - Actor 学习率: {lr_actor}")
    print(f"  - Critic 学习率: {lr_critic}")
    print(f"\n开始训练...\n")
    
    for ep in range(max_episodes):
        # 收集经验
        buffer.clear()
        total_reward, length = collect_experience(
            env, actor, buffer, device, 
            max_steps=max_steps_per_episode, 
            seed=seed + ep
        )
        
        # PPO 更新
        actor_loss, critic_loss = ppo_update(
            env, actor, critic, buffer, opt_actor, opt_critic, device,
            gamma=gamma, eps_clip=eps_clip, k_epochs=k_epochs
        )
        
        # 记录
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        recent_rewards.append(total_reward)
        
        # 打印进度
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {ep + 1:4d}/{max_episodes}  "
                  f"reward={total_reward:8.2f}  "
                  f"avg_reward(最近100)={avg_reward:8.2f}  "
                  f"steps={length:3d}  "
                  f"actor_loss={actor_loss:.4f}  "
                  f"critic_loss={critic_loss:.4f}")
        
        # 保存模型
        if (ep + 1) % save_frequency == 0:
            model_path = os.path.join(model_dir, f"ppo_pendulum_episode_{ep + 1}.pth")
            save_model(actor, critic, opt_actor, opt_critic, model_path)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "ppo_pendulum_final.pth")
    save_model(actor, critic, opt_actor, opt_critic, final_model_path)
    
    # 画训练曲线
    plot_training_curve(episode_rewards, save_path="./training_curve.png")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最终平均奖励(最近100个episode): {np.mean(recent_rewards):.2f}")
    print(f"模型已保存到: {model_dir}")
    print("=" * 50)
    
    env.close()
    
    return actor, critic, episode_rewards


# =============================================================================
# 4. 主函数
# =============================================================================


if __name__ == "__main__":
    train(
        max_episodes=500,
        max_steps_per_episode=200,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=5,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_dim=64,
        save_frequency=100,
        model_dir="./models",
        seed=42,
    )
