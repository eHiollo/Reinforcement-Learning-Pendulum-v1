"""
第九步：评估与可视化

加载训练好的模型，在环境里跑几个 episode：
- 可选渲染（看倒立摆动画）
- 可视化状态、动作、奖励轨迹
- 分析智能体的表现
- 生成视频/GIF

运行：python step_09_evaluate.py
生成视频：python step_09_evaluate.py --video
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from step_02_env_wrapper import PendulumWrapper
from step_05_critic import Critic
from step_06_collect_experience import PPOActor, actor_select_action_with_log_prob

# 视频录制支持
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# =============================================================================
# 1. 加载模型
# =============================================================================


def load_model(actor, critic, filepath, device):
    """
    加载 Actor 和 Critic 的模型状态（评估时不需要优化器）
    
    Returns:
        checkpoint: 完整的checkpoint字典（可用于查看训练信息）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # 打印训练信息（如果有）
    if 'episode' in checkpoint:
        print(f"模型来自 Episode {checkpoint['episode']}")
    if 'episode_rewards' in checkpoint and checkpoint['episode_rewards']:
        rewards = checkpoint['episode_rewards']
        print(f"训练时最后100回合平均奖励: {np.mean(rewards[-100:]):.2f}")
    
    print(f"模型已从 {filepath} 加载")
    return checkpoint


# =============================================================================
# 2. 评估：用确定性策略跑几个 episode
# =============================================================================


def evaluate_episode(env, actor, device, max_steps=200, seed=None, render=False):
    """
    用当前 Actor 跑一个 episode（确定性策略：直接用 mean，不采样）
    
    返回: (states_history, actions_history, rewards_history, total_reward, length)
    """
    state, _ = env.reset(seed=seed)  # Gymnasium返回 (obs, info)
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

        env.render()
        
        if render:
            env.render()
        
        state = next_state
        if done:
            break
    
    return states_history, actions_history, rewards_history, total_reward, len(rewards_history)


def record_video_episode(env, actor, device, max_steps=200, seed=None, video_path="./pendulum_demo.gif"):
    """
    录制一个 episode 的视频/GIF
    """
    if not HAS_IMAGEIO:
        print("需要安装 imageio: pip install imageio")
        print("运行: pip install imageio imageio-ffmpeg")
        return None
    
    import imageio  # 在函数内部导入，避免未定义错误
    import gymnasium as gym
    
    # 创建一个新的环境用于录制（rgb_array模式）
    raw_env = gym.make("Pendulum-v1", render_mode="human")
    
    frames = []
    state, _ = raw_env.reset(seed=seed)
    
    # 录制第一帧
    frames.append(raw_env.render())
    
    for step in range(max_steps):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mean, log_std = actor(state_tensor)
            action_unbounded = mean
            action = torch.tanh(action_unbounded) * 2.0
            action = action.squeeze(0).cpu().numpy()
        
        next_state, reward, terminated, truncated, _ = raw_env.step(action)
        done = terminated or truncated
        
        # 录制帧
        frames.append(raw_env.render())
        
        state = next_state
        if done:
            break
    
    raw_env.close()
    
    # 保存为 GIF
    if video_path.endswith('.gif'):
        imageio.mimsave(video_path, frames, fps=30, loop=0)
    else:
        # 保存为 MP4
        imageio.mimsave(video_path, frames, fps=30)
    
    print(f"视频已保存到: {video_path}")
    return frames


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
    model_path="models/ppo_pendulum_episode_2300.pth",
    num_episodes=10,
    max_steps_per_episode=200,
    render=True,
    seed=42,
    hidden_dim=64,  # 需要与训练时一致
):
    """
    评估训练好的模型
    """
    print("=" * 50)
    print("第九步：评估与可视化")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    render_mode = "human" if render else "rgb_array"
    env = PendulumWrapper("Pendulum-v1", render_mode=render_mode)
    
    # 创建网络（hidden_dim需要与训练时一致）
    actor = PPOActor(env.state_dim, env.action_dim, hidden=hidden_dim).to(device)
    critic = Critic(env.state_dim, hidden=hidden_dim).to(device)
    
    # 加载模型（评估时不需要优化器）
    checkpoint = load_model(actor, critic, model_path, device)
    
    # 设置为评估模式
    actor.eval()
    critic.eval()
    
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
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估PPO训练的模型')
    parser.add_argument('--model', '-m', type=str, default="models/ppo_pendulum_episode_2300.pth",
                        help='模型文件路径（默认使用最新的模型）')
    parser.add_argument('--episodes', '-n', type=int, default=20,
                        help='评估的episode数量（默认10）')
    parser.add_argument('--render', '-r', action='store_true',
                        help='是否渲染动画（需要图形界面）')
    parser.add_argument('--video', '-v', action='store_true',
                        help='录制视频/GIF')
    parser.add_argument('--video-path', type=str, default='./pendulum_demo.gif',
                        help='视频保存路径（默认 ./pendulum_demo.gif）')
    parser.add_argument('--seed', '-s', type=int, default=3,
                        help='随机种子（默认42）')
    args = parser.parse_args()
    
    # 如果没有指定模型，自动查找最新的模型
    models_dir = "models"
    if args.model is None:
        if os.path.exists(models_dir):
            available = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if available:
                # 按episode数字排序，选择最新的
                def get_episode_num(filename):
                    try:
                        # 格式: ppo_pendulum_episode_XXX.pth
                        return int(filename.split('_')[-1].replace('.pth', ''))
                    except:
                        return 0
                available.sort(key=get_episode_num, reverse=True)
                model_path = os.path.join(models_dir, available[0])
                print(f"自动选择最新模型: {model_path}")
            else:
                print(f"models目录中没有找到模型文件")
                model_path = None
        else:
            print(f"models目录不存在")
            model_path = None
    else:
        model_path = args.model
    
    if model_path and os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = PPOActor(3, 1, hidden=64).to(device)
        critic = Critic(3, hidden=64).to(device)
        load_model(actor, critic, model_path, device)
        actor.eval()
        
        # 默认生成视频（WSL环境下无法直接显示图形界面）
        if args.video or not args.render:
            print("=" * 50)
            print("录制视频（WSL环境下推荐方式）...")
            print("=" * 50)
            try:
                record_video_episode(None, actor, device, max_steps=200, 
                                   seed=args.seed, video_path=args.video_path)
                abs_path = os.path.abspath(args.video_path)
                print(f"\n✅ 视频已保存到: {abs_path}")
                print(f"\n在 Windows 中查看视频的方法：")
                print(f"  1. 在文件资源管理器地址栏输入：\\\\wsl$\\Ubuntu{abs_path}")
                print(f"  2. 或者直接复制到Windows：cp {abs_path} /mnt/c/Users/你的用户名/Desktop/")
            except Exception as e:
                print(f"⚠️  录制视频失败: {e}")
                print("   提示: 运行 'pip install imageio imageio-ffmpeg' 安装依赖")
        
        # 运行评估
        evaluate(
            model_path=model_path,
            num_episodes=args.episodes,
            max_steps_per_episode=200,
            render=args.render,
            seed=args.seed,
            hidden_dim=64,  # 需要与训练时一致
        )
    else:
        if model_path:
            print(f"模型文件不存在: {model_path}")
        print("请先运行 step_08_train.py 训练模型")
        # 列出可用的模型文件
        if os.path.exists(models_dir):
            available = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if available:
                print(f"\n可用的模型文件:")
                for f in sorted(available):
                    print(f"  - {models_dir}/{f}")
