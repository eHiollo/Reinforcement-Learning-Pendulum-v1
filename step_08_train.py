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
# TensorBoard支持
# 注意：PyTorch 1.1+ 支持 torch.utils.tensorboard
# 如果导入失败，请确保安装了 tensorboard: pip install tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:
    # 备用导入方式
    SummaryWriter = None  # type: ignore
from datetime import datetime

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


def save_model(actor, critic, opt_actor, opt_critic, filepath, episode=None, episode_rewards=None, episode_lengths=None):
    """
    保存 Actor 和 Critic 的模型和优化器状态
    
    Args:
        episode: 当前episode数（用于恢复训练）
        episode_rewards: 历史reward列表（用于恢复训练曲线）
        episode_lengths: 历史length列表（用于恢复训练曲线）
    """
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': opt_actor.state_dict(),
        'critic_optimizer_state_dict': opt_critic.state_dict(),
    }
    if episode is not None:
        checkpoint['episode'] = episode
    if episode_rewards is not None:
        checkpoint['episode_rewards'] = episode_rewards
    if episode_lengths is not None:
        checkpoint['episode_lengths'] = episode_lengths
    torch.save(checkpoint, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(actor, critic, opt_actor, opt_critic, filepath, device):
    """
    从checkpoint加载模型和优化器状态
    
    Returns:
        episode: 保存时的episode数（如果存在）
        episode_rewards: 历史reward列表（如果存在）
        episode_lengths: 历史length列表（如果存在）
    """
    checkpoint = torch.load(filepath, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    opt_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    opt_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    episode = checkpoint.get('episode', None)
    episode_rewards = checkpoint.get('episode_rewards', None)
    episode_lengths = checkpoint.get('episode_lengths', None)
    
    print(f"模型已从 {filepath} 加载")
    if episode is not None:
        print(f"  恢复episode: {episode}")
    if episode_rewards is not None:
        print(f"  恢复历史reward: {len(episode_rewards)} 个episode")
    if episode_lengths is not None:
        print(f"  恢复历史length: {len(episode_lengths)} 个episode")
    
    return episode, episode_rewards, episode_lengths


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
    use_tensorboard=True,
    log_dir="./runs",
    num_envs=32,  # 并行环境数量，RTX 3060推荐16-64
    checkpoint_path=None,  # 从checkpoint恢复训练，例如："./models/ppo_pendulum_episode_100.pth"
):
    """
    完整训练循环
    
    Args:
        use_tensorboard: 是否使用TensorBoard记录训练过程
        log_dir: TensorBoard日志保存目录
    """
    # 设置随机种子/ 让numpy和torch给随机数都是根据seed=42来的
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 【新增】初始化TensorBoard
    writer = None
    if use_tensorboard:
        if SummaryWriter is None:
            print("警告：TensorBoard未安装，跳过TensorBoard记录")
            print("安装方法: pip install tensorboard")
            use_tensorboard = False
        else:
            # 创建带时间戳的日志目录，方便区分不同训练运行
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"ppo_pendulum_{timestamp}")
            os.makedirs(log_path, exist_ok=True)
            writer = SummaryWriter(log_dir=log_path)
            print(f"TensorBoard日志保存到: {log_path}")
            print(f"查看训练过程: tensorboard --logdir {log_dir}")
        
        # 记录超参数到TensorBoard（方便对比不同配置）
        writer.add_hparams(
            {
                'max_episodes': max_episodes,
                'max_steps_per_episode': max_steps_per_episode,
                'gamma': gamma,
                'eps_clip': eps_clip,
                'k_epochs': k_epochs,
                'lr_actor': lr_actor,
                'lr_critic': lr_critic,
                'hidden_dim': hidden_dim,
                'gae_lambda': 0.95,
                'num_envs': num_envs,
            },
            {}
        )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 多环境并行
    from gymnasium.vector import SyncVectorEnv
    def make_env():
        return lambda: PendulumWrapper("Pendulum-v1")

    # 【环境数量说明】
    # - RTX 3060 (12GB): 推荐 16-64 个环境
    #   * 16个: 显存占用低，训练稳定，适合调试
    #   * 32个: 平衡显存和速度（当前默认）
    #   * 64个: 速度快，但显存占用高，需要监控
    # - 环境越多，采样速度越快，但显存占用也越高
    # - 使用 test_env_count.py 可以测试最优配置
    env_fns = [make_env() for _ in range(num_envs)]
    envs = SyncVectorEnv(env_fns)
    print(f"并行环境数量: {num_envs}")
    
    # 网络和优化器
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    actor = PPOActor(obs_dim, act_dim, hidden=hidden_dim).to(device)
    critic = Critic(obs_dim, hidden=hidden_dim).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    
    # 【新增】从checkpoint恢复训练
    start_episode = 0
    episode_rewards = []
    episode_lengths = []
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"\n从checkpoint恢复训练: {checkpoint_path}")
        loaded_episode, loaded_rewards, loaded_lengths = load_model(actor, critic, opt_actor, opt_critic, checkpoint_path, device)
        if loaded_episode is not None:
            start_episode = loaded_episode
            print(f"从episode {start_episode} 继续训练")
        if loaded_rewards is not None:
            episode_rewards = loaded_rewards
            if loaded_lengths is not None:
                episode_lengths = loaded_lengths
            else:
                episode_lengths = [200] * len(loaded_rewards)  # 如果没有保存lengths，假设都是200步
            print(f"恢复历史数据: {len(episode_rewards)} 个episode")
    else:
        if checkpoint_path is not None:
            print(f"警告：checkpoint文件不存在: {checkpoint_path}，从头开始训练")
    
    buffer = ExperienceBuffer()

    def collect_experience_vector(envs, actor, buffer, device, max_steps=200):
        """
        用 vector 环境并行采样 num_envs 条轨迹，存到 buffer。
        返回每个环境的总奖励和步数。
        """
        num_envs = envs.num_envs
        states, _ = envs.reset()
        total_rewards = np.zeros(num_envs)
        lengths = np.zeros(num_envs, dtype=int)
        for step in range(max_steps):
            states_tensor = torch.from_numpy(states).float().to(device)
            with torch.no_grad():
                mean, log_std = actor(states_tensor)
                std = torch.exp(log_std)
                normal = torch.distributions.Normal(mean, std)
                action_unbounded = normal.sample()
                log_prob = normal.log_prob(action_unbounded).sum(dim=-1)
                action = torch.tanh(action_unbounded) * 2.0
                log_prob -= torch.log(1 - torch.tanh(action_unbounded).pow(2) + 1e-6).sum(dim=-1)
                log_prob -= np.log(2.0)
            actions_np = action.cpu().numpy()
            next_states, rewards, terminated, truncated, infos = envs.step(actions_np)
            dones = np.logical_or(terminated, truncated)
            #rollout 存进 buffer（包含next_states用于GAE）
            for i in range(num_envs):
                buffer.add(states[i], actions_np[i], rewards[i], next_states[i], dones[i], log_prob[i].item())
                total_rewards[i] += rewards[i]
                lengths[i] += 1
            states = next_states
            if np.all(dones):
                break
        return total_rewards, lengths
    
    # 记录训练过程（如果从checkpoint恢复，这些列表可能已经有数据）
    recent_rewards = deque(maxlen=100)  # 最近 100 个 episode 的平均
    # 如果从checkpoint恢复，填充recent_rewards
    if len(episode_rewards) > 0:
        for r in episode_rewards[-100:]:
            recent_rewards.append(r)
    
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
    
    for ep in range(start_episode, max_episodes):
        buffer.clear()
        total_rewards, lengths = collect_experience_vector(
            envs, actor, buffer, device, max_steps=max_steps_per_episode
        )
        actor_loss, critic_loss, entropy, kl_div, mean_advantage = ppo_update(
            envs, actor, critic, buffer, opt_actor, opt_critic, device,
            gamma=gamma, eps_clip=eps_clip, k_epochs=k_epochs,
            gae_lambda=0.95, use_gae=True, num_envs=num_envs,
            value_coef=1.0, entropy_coef=0.01, max_grad_norm=0.5  # entropy_coef增加，防止策略变得过于确定
        )
        

        # 记录
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(lengths)
        episode_rewards.append(float(mean_reward))
        episode_lengths.append(int(mean_length))
        recent_rewards.append(mean_reward)
        
        # 【新增】记录到TensorBoard
        if writer is not None:
            # 主要指标
            writer.add_scalar('Reward/Episode_Reward', mean_reward, ep + 1)
            writer.add_scalar('Reward/Average_Reward_100', np.mean(recent_rewards), ep + 1)
            writer.add_scalar('Reward/Max_Reward', np.max(total_rewards), ep + 1)
            writer.add_scalar('Reward/Min_Reward', np.min(total_rewards), ep + 1)
            
            # Episode长度
            writer.add_scalar('Episode/Length', mean_length, ep + 1)
            
            # Loss指标
            writer.add_scalar('Loss/Actor_Loss', actor_loss, ep + 1)
            writer.add_scalar('Loss/Critic_Loss', critic_loss, ep + 1)
            
            # 策略指标
            writer.add_scalar('Policy/Entropy', entropy, ep + 1)
            writer.add_scalar('Policy/KL_Divergence', kl_div, ep + 1)
            writer.add_scalar('Policy/Mean_Advantage', mean_advantage, ep + 1)
            
            # 统计信息（每10个episode记录一次详细统计）
            if (ep + 1) % 10 == 0:
                writer.add_scalar('Stats/Reward_Std', np.std(total_rewards), ep + 1)
                writer.add_scalar('Stats/Length_Std', np.std(lengths), ep + 1)
        
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {ep + 1:4d}/{max_episodes}  "
                  f"mean_reward={mean_reward:8.2f}  "
                  f"avg_reward(最近100)={avg_reward:8.2f}  "
                  f"mean_steps={mean_length:3.0f}  "
                  f"actor_loss={actor_loss:.4f}  "
                  f"critic_loss={critic_loss:.4f}  "
                  f"entropy={entropy:.4f}  "
                  f"kl={kl_div:.4f}")
        if (ep + 1) % save_frequency == 0:
            model_path = os.path.join(model_dir, f"ppo_pendulum_episode_{ep + 1}.pth")
            save_model(actor, critic, opt_actor, opt_critic, model_path, 
                      episode=ep + 1, episode_rewards=episode_rewards, episode_lengths=episode_lengths)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "ppo_pendulum_final.pth")
    save_model(actor, critic, opt_actor, opt_critic, final_model_path,
              episode=max_episodes, episode_rewards=episode_rewards, episode_lengths=episode_lengths)
    
    # 画训练曲线
    plot_training_curve(episode_rewards, save_path="./training_curve.png")
    
    # 【新增】关闭TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"\nTensorBoard日志已保存，使用以下命令查看：")
        print(f"  tensorboard --logdir {log_dir}")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最终平均奖励(最近100个episode): {np.mean(recent_rewards):.2f}")
    print(f"模型已保存到: {model_dir}")
    print("=" * 50)
    
    envs.close()
    
    return actor, critic, episode_rewards


# =============================================================================
# 4. 主函数
# =============================================================================


if __name__ == "__main__":
    train(
        max_episodes=5000,
        max_steps_per_episode=200,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,  # 增加更新轮数，配合更低的学习率
        lr_actor=1e-4,   # Actor学习率（降低，防止策略崩溃）
        lr_critic=3e-4,  # Critic学习率
        hidden_dim=64,
        save_frequency=100,
        model_dir="./models",
        seed=42,
        num_envs=64,  # RTX 3060推荐：16-64
        # 当前64个环境 × 200步 = 12800步/rollout，符合PPO最佳实践（推荐2048-4096步）
        # 如果想更稳定（显存紧张）：num_envs=32（6400步/rollout，也足够）
        # 如果想更快：num_envs=64（当前配置）
        checkpoint_path="models/ppo_pendulum_episode_1000.pth",  # 【重要】从头开始训练！之前的checkpoint是用错误的GAE计算的
    )
