# PPO 强化学习项目 - 倒立摆控制

使用 PPO (Proximal Policy Optimization) 算法训练智能体控制倒立摆环境。

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python step_08_train.py
```

### 评估模型

```bash
# 评估最新模型
python step_09_evaluate.py

# 指定模型路径
python step_09_evaluate.py --model models/ppo_pendulum_episode_1000.pth

# 生成视频
python step_09_evaluate.py --video
```

### 查看训练过程

```bash
tensorboard --logdir ./runs --port 6006
```

然后在浏览器访问 `http://localhost:6006`

## 📁 项目结构

```
.
├── step_01_explore_env.py          # 认识环境
├── step_02_env_wrapper.py           # 环境封装
├── step_04_actor.py                # Actor 网络
├── step_05_critic.py               # Critic 网络
├── step_06_collect_experience.py   # 经验收集
├── step_07_ppo_update.py           # PPO 更新逻辑
├── step_08_train.py                # 完整训练循环
├── step_09_evaluate.py             # 模型评估
├── requirements.txt                # 依赖列表
├── models/                         # 保存的模型
└── runs/                           # TensorBoard 日志
```

## 🎯 主要特性

- ✅ PPO 算法实现（带 GAE）
- ✅ Actor-Critic 架构
- ✅ 并行环境训练（加速数据收集）
- ✅ TensorBoard 可视化
- ✅ 模型检查点保存/恢复
- ✅ 训练稳定性保障（梯度裁剪、NaN 检测等）

## 📚 学习资源

- `强化学习完整知识点讲解.md` - 详细的理论讲解
- `强化学习快速参考.md` - 快速查阅公式和概念

## 🔧 主要超参数

- `gamma = 0.99` - 折扣因子
- `gae_lambda = 0.95` - GAE 参数
- `eps_clip = 0.2` - PPO 裁剪范围
- `k_epochs = 4` - 多轮更新次数
- `lr_actor = 1e-4` - Actor 学习率
- `lr_critic = 3e-4` - Critic 学习率
- `num_envs = 32` - 并行环境数量

## 📝 依赖

- `gymnasium` - 强化学习环境
- `torch` - 深度学习框架
- `numpy` - 数值计算
- `matplotlib` - 可视化
- `tensorboard` - 训练监控
- `imageio` - 视频生成

## 🎓 学习路径

按照 step_01 到 step_09 的顺序逐步学习，每个文件都有详细注释。

## 📄 许可证

MIT License

