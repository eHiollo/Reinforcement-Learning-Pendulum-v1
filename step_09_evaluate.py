"""
第九步：评估与可视化

加载训练好的模型，在环境里跑几个 episode：
- 可选渲染（看倒立摆动画）
- 可视化状态、动作、奖励轨迹
- 分析智能体的表现

运行：python step_09_evaluate.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from step_02_env_wrapper import PendulumWrapper
from step_05_critic import Critic
from step_06_collect_experience import PPOActor, actor_select_action_with_log_prob


# =============================================================================
# 1. 加载模型
# =============================================================================


def load_model(actor, critic, opt_actor, opt_critic, filepath, device):
    """加载 Actor 和 Critic 的模型和优化器状态"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    opt_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    opt_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    print(f"模型已从 {filepath} 加载")


# =============================================================================
# 2. 评估：用确定性策略跑几个 episode
# =============================================================================


def evaluate_episode(env, actor, device, max_steps=200, seed=None, render=False):
    """
    用当前 Actor 跑一个 episode（确定性策略：直接用 mean，不采样）
    
    返回: (states_history, actions_history, rewards_history, total_reward, length)
    """
    state = env.reset(seed=seed)
    states_history = [state.copy()]
    actions_history = []
    rewards_history = []
    total_reward = 0.0
    
    for step in range(max_steps):
        # 确定性策略：直接用 mean，不采样
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mean, log_std = actor(state_tensor)
            # 直接用 mean，限制到 [-2, 2]
            action_unbounded = mean
            action = torch.tanh(action_unbounded) * 2.0
            action = action.squeeze(0).cpu().numpy()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        states_history.append(next_state.copy())
        actions_history.append(action.copy())
        rewards_history.append(reward)
        total_reward += reward
        
        if render:
            env.render()
        
        state = next_state
        if done:
            break
    
    return states_history, actions_history, rewards_history, total_reward, len(rewards_history)


# =============================================================================
# 3. 可视化一个 episode 的轨迹
# =============================================================================


def visualize_episode(states_history, actions_history, rewards_history, 
                      save_path="./episode_visualization.png"):
    """
    可视化一个 episode 的状态、动作、奖励轨迹
    """
    states_array = np.array(states_history)
    actions_array = np.array(actions_history)
    rewards_array = np.array(rewards_history)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    time_steps = np.arange(len(states_array))
    action_time_steps = np.arange(len(actions_array))
    reward_time_steps = np.arange(len(rewards_array))
    
    # 状态轨迹
    axes[0].plot(time_steps, states_array[:, 0], label='cos(θ)', linewidth=2, alpha=0.7)
    axes[0].plot(time_steps, states_array[:, 1], label='sin(θ)', linewidth=2, alpha=0.7)
    axes[0].plot(time_steps, states_array[:, 2], label='θ_dot', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('State Value')
    axes[0].set_title('State Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 动作轨迹
    axes[1].plot(action_time_steps, actions_array[:, 0], 'r-', linewidth=2, label='Action')
    axes[1].axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='Action Limit (+2)')
    axes[1].axhline(y=-2.0, color='g', linestyle='--', alpha=0.5, label='Action Limit (-2)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Action Value')
    axes[1].set_title('Action Trajectory')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 奖励轨迹
    axes[2].plot(reward_time_steps, rewards_array, 'b-', linewidth=2, label='Reward')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Reward Trajectory')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Episode 可视化已保存到: {save_path}")
    plt.close()


# =============================================================================
# 4. 主评估函数
# =============================================================================


def evaluate(
    model_path="./models/ppo_pendulum_final.pth",
    num_episodes=10,
    max_steps_per_episode=200,
    render=False,
    seed=42,
):
    """
    评估训练好的模型
    """
    print("=" * 50)
    print("第九步：评估与可视化")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PendulumWrapper("Pendulum-v1", render_mode="human" if render else None)
    
    actor = PPOActor(env.state_dim, env.action_dim, hidden=64).to(device)
    critic = Critic(env.state_dim, hidden=64).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    
    # 加载模型
    load_model(actor, critic, opt_actor, opt_critic, model_path, device)
    
    print(f"\n评估配置:")
    print(f"  - Episode 数: {num_episodes}")
    print(f"  - 每 episode 最大步数: {max_steps_per_episode}")
    print(f"  - 渲染: {'是' if render else '否'}")
    print(f"\n开始评估...\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        states, actions, rewards, total_reward, length = evaluate_episode(
            env, actor, device, 
            max_steps=max_steps_per_episode, 
            seed=seed + ep,
            render=render
        )
        
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        
        print(f"Episode {ep + 1:2d}/{num_episodes}: "
              f"reward={total_reward:8.2f}  steps={length:3d}")
        
        # 可视化第一个 episode
        if ep == 0:
            visualize_episode(states, actions, rewards, 
                             save_path=f"./episode_0_visualization.png")
    
    # 统计信息
    print("\n" + "=" * 50)
    print("评估结果统计:")
    print(f"  - 平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  - 平均步数: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  - 最大奖励: {np.max(episode_rewards):.2f}")
    print(f"  - 最小奖励: {np.min(episode_rewards):.2f}")
    print("=" * 50)
    
    env.close()
    
    return episode_rewards, episode_lengths


# =============================================================================
# 5. 主函数
# =============================================================================


if __name__ == "__main__":
    # 评估模型（需要先训练模型）
    model_path = "models/ppo_pendulum_episode_400.pth"
    
    if os.path.exists(model_path):
        evaluate(
            model_path=model_path,
            num_episodes=10,
            max_steps_per_episode=200,
            render=False,  # 设为 True 可以看到动画（需要图形界面）
            seed=42,
        )
    else:
        print(f"模型文件不存在: {model_path}")
        print("请先运行 step_08_train.py 训练模型")
