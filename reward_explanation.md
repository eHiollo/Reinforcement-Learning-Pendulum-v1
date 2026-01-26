# Reward 计算说明

## 1. Reward 是环境自动计算的

在我们的代码里，**reward 是 Gymnasium 的 Pendulum 环境自动计算的**，我们不需要自己写。

当你调用：
```python
next_state, reward, terminated, truncated, info = env.step(action)
```

环境内部会自动：
1. 根据当前状态 `state` 和动作 `action`，用物理公式计算下一个状态
2. **根据状态和动作计算 reward**
3. 返回给你

---

## 2. Pendulum 环境的 Reward 公式

Gymnasium 的 `Pendulum-v1` 环境的 reward 公式是：

\[
\text{reward} = -(\theta^2 + 0.1 \cdot \dot{\theta}^2 + 0.001 \cdot u^2)
\]

其中：
- **\(\theta\)**：摆的角度（从竖直向上算起，单位：弧度）
  - \(\theta = 0\) 表示竖直向上（目标状态）
  - 可以从 state 计算：\(\theta = \text{atan2}(\sin(\theta), \cos(\theta))\)，或直接用 `state[0]` 和 `state[1]` 反推
- **\(\dot{\theta}\)**：角速度，就是 `state[2]`（θ_dot）
- **\(u\)**：施加的力矩，就是 `action[0]`

### 公式含义

- **\(\theta^2\)**：惩罚角度偏离竖直（角度越大，惩罚越大）
- **\(0.1 \cdot \dot{\theta}^2\)**：惩罚角速度（角速度越大，惩罚越大）
- **\(0.001 \cdot u^2\)**：惩罚动作幅度（动作越大，惩罚越大，鼓励用小的力）

**Reward 是负数**，因为：
- 越小越好（接近 0 最好）
- 当摆竖直向上（\(\theta = 0\)）、静止（\(\dot{\theta} = 0\)）、不施力（\(u = 0\)）时，reward = 0（最大值）

---

## 3. 为什么 Reward 是负数？

在强化学习中，reward 可以是正数或负数：
- **正数**：奖励，越大越好（如游戏得分）
- **负数**：惩罚，越小越好（如这里的成本函数）

Pendulum 用负数是因为：
- 目标是最小化「角度偏差 + 角速度 + 动作成本」
- 等价于最大化「负的（角度偏差 + 角速度 + 动作成本）」
- 所以 reward = -成本，reward 越大（越接近 0）越好

---

## 4. 在代码里怎么看 Reward？

### 第一步（step_01_explore_env.py）

```python
next_obs, reward, terminated, truncated, info = env.step(action)
print("  本步奖励 reward =", reward)  # 会打印一个负数，如 -1.234
```

### 收集经验时（step_06_collect_experience.py）

```python
next_state, reward, terminated, truncated, _ = env.step(action)
buffer.add(state, action, reward, done, log_prob)  # reward 被存起来
```

### PPO 更新时（step_07_ppo_update.py）

```python
returns = compute_returns(rewards, dones, gamma)  # 用 rewards 计算折扣回报
```

---

## 5. 如果想自定义 Reward 怎么办？

如果你想**修改 reward 函数**（比如改变权重、加额外奖励），有几种方法：

### 方法一：包装环境（Wrapper）

创建一个 Wrapper，在 `step()` 里修改 reward：

```python
class CustomRewardWrapper:
    def __init__(self, env):
        self.env = env
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # 自定义 reward
        # 例如：如果角度接近 0，给额外奖励
        theta = np.arctan2(next_state[1], next_state[0])  # 从 cos, sin 反推角度
        if abs(theta) < 0.1:  # 角度很小
            reward += 1.0  # 额外奖励
        
        return next_state, reward, terminated, truncated, info
```

### 方法二：修改 PendulumWrapper

在 `step()` 方法里修改返回的 reward：

```python
def step(self, action):
    next_obs, reward, terminated, truncated, info = self.env.step(action)
    next_state = np.array(next_obs, dtype=np.float32)
    
    # 在这里修改 reward
    # reward = your_custom_reward_function(next_state, action, reward)
    
    return next_state, float(reward), bool(terminated), bool(truncated), info
```

---

## 6. 总结

| 问题 | 答案 |
|------|------|
| **Reward 是谁算的？** | Gymnasium 的 Pendulum 环境自动计算 |
| **公式是什么？** | `reward = -(θ² + 0.1·θ_dot² + 0.001·u²)` |
| **为什么是负数？** | 表示成本，越小越好（接近 0 最好） |
| **我们需要写吗？** | 不需要，环境自动算好返回给我们 |
| **能自定义吗？** | 可以，用 Wrapper 或修改 `step()` 方法 |

---

**关键点**：在强化学习中，**reward 函数是环境的一部分**，定义了任务的目标。我们只需要：
1. 调用 `env.step(action)` 拿到 reward
2. 用 reward 训练智能体（PPO 算法）

不需要自己实现 reward 的计算（除非你想自定义）。
