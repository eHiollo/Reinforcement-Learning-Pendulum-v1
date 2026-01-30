"""
第二步：封装环境

目标：写一个 PendulumWrapper，把 reset / step 包成统一接口，
      输出统一用 float32、形状固定，方便后面接「策略网络」。

运行：python step_02_env_wrapper.py
"""

import gymnasium as gym
import numpy as np


# =============================================================================
# 为什么要封装？
# =============================================================================
# 1. 统一格式：state 永远 np.float32，action 永远先转成数组再传进去
# 2. 方便取 state_dim、action_dim：后面建神经网络时要用到
# 3. 以后换环境（比如 MuJoCo 倒立摆）只改这一层，训练代码不用动
# =============================================================================


class PendulumWrapper:
    """
    倒立摆环境封装。
    对外只暴露：reset()、step(action)、close()，以及 state_dim、action_dim。
    """

    def __init__(self, env_name: str = "Pendulum-v1", render_mode: str = "human"):
        """
        创建环境，并记下 state_dim、action_dim。

        env_name: 环境名，默认 Pendulum-v1
        render_mode: None 不渲染；'human' 可弹窗看动画（评估时用）
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # 兼容 vector 环境
        self.metadata = self.env.metadata
        self.render_mode = getattr(self.env, 'render_mode', render_mode)
        self.metadata = self.env.metadata

        # 后面建 Actor/Critic 网络时要用的维度
        self.state_dim = self.observation_space.shape[0]   # 3
        self.action_dim = self.action_space.shape[0]       # 1

    def reset(self, seed: int = None, **kwargs):
        """
        重置环境，返回 (obs, info) tuple，兼容 vector 环境。

        seed: 随机种子，复现用。
        kwargs: 其他 gymnasium 可能传递的参数（如 options）。
        """
        obs, info = self.env.reset(seed=seed, **kwargs)
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        """
        执行一步。

        action: 一维动作，可以是 float、list 或 shape=(1,) 的数组，我们会转成数组。

        返回：
            next_state: np.float32, shape=(3,)
            reward: float
            terminated: bool，episode 是否「自然结束」
            truncated: bool，是否「被截断」（如超时）
            info: dict，额外信息（一般训练不用）
        """
        # 统一转成 (1,) 的 float 数组，避免 shape 不一致
        action = np.atleast_1d(np.asarray(action, dtype=np.float32))

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_state = np.array(next_obs, dtype=np.float32)

        return next_state, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        """关闭环境，释放资源。"""
        self.env.close()


# =============================================================================
# 小测试：用 Wrapper 跑几步，跟 step_01 逻辑一样，只是接口换了
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("第二步：封装环境 —— 用 PendulumWrapper 跑几步")
    print("=" * 50)

    env = PendulumWrapper("Pendulum-v1")
    print(f"\nstate_dim = {env.state_dim}, action_dim = {env.action_dim}\n")

    # reset：注意现在只返回 state，没有 info
    state = env.reset(seed=42)
    print("reset(seed=42) -> state =", state)

    # 不施力，执行一步
    next_state, reward, terminated, truncated, info = env.step([0.0])
    print("\nstep([0.0]) ->")
    print("  next_state =", next_state)
    print("  reward     =", reward)
    print("  done       =", terminated or truncated)

    # 随机动作跑 5 步
    print("\n随机动作跑 5 步：")
    state = env.reset(seed=123)
    for t in range(5):
        action = env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        print(f"  t={t+1}  action={action[0]:7.3f}  reward={reward:7.3f}  "
              f"cos={next_state[0]:6.3f}  sin={next_state[1]:6.3f}  "
              f"θ_dot={next_state[2]:6.3f}  done={done}")
        state = next_state

    env.close()
    print("\n" + "=" * 50)
    print("第二步完成！之后训练时一律用 PendulumWrapper，不再直接调 gym。")
    print("=" * 50)
