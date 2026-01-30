# 📊 TensorBoard 启动指南

## 🚀 快速启动

### 方法1：在 WSL 中启动（推荐）

```bash
# 进入项目目录
cd /home/allen/projects/RL

# 启动 TensorBoard（查看所有训练运行）
tensorboard --logdir ./runs --port 6006

# 或者指定特定的一次训练运行
tensorboard --logdir ./runs/ppo_pendulum_20260130_140435 --port 6006
```

**然后在 Windows 浏览器中访问**:
```
http://localhost:6006
```

### 方法2：后台运行

```bash
# 后台运行 TensorBoard
nohup tensorboard --logdir ./runs --port 6006 > tensorboard.log 2>&1 &

# 查看进程
ps aux | grep tensorboard

# 停止 TensorBoard（找到进程ID后）
kill <进程ID>
```

### 方法3：指定主机和端口

```bash
# 允许外部访问（如果需要）
tensorboard --logdir ./runs --host 0.0.0.0 --port 6006
```

---

## 📁 日志目录结构

```
./runs/
├── ppo_pendulum_20260130_033029/  # 训练运行1
├── ppo_pendulum_20260130_033236/  # 训练运行2
├── ppo_pendulum_20260130_034718/  # 训练运行3
└── ...
```

每次训练都会创建一个带时间戳的新目录。

---

## 🔍 查看特定训练运行

如果你想查看最新的一次训练：

```bash
# 找到最新的日志目录
LATEST_RUN=$(ls -td ./runs/ppo_pendulum_* | head -1)
tensorboard --logdir $LATEST_RUN --port 6006
```

或者直接指定目录：

```bash
tensorboard --logdir ./runs/ppo_pendulum_20260130_140435 --port 6006
```

---

## 📊 TensorBoard 中可以看到的指标

根据你的 `step_08_train.py`，TensorBoard 会记录：

1. **Reward/Episode**: 每个 episode 的奖励
2. **Reward/MovingAverage**: 移动平均奖励
3. **Loss/Actor**: Actor 损失
4. **Loss/Critic**: Critic 损失
5. **Policy/Entropy**: 策略熵（探索程度）
6. **Policy/KLDivergence**: KL 散度（策略变化）
7. **Advantage/Mean**: 平均优势
8. **Episode/Length**: Episode 长度

---

## 🛠️ 常见问题

### 问题1：端口被占用

```bash
# 使用其他端口
tensorboard --logdir ./runs --port 6007
```

### 问题2：找不到 tensorboard 命令

```bash
# 安装 TensorBoard
pip install tensorboard

# 或者如果使用 conda
conda install tensorboard
```

### 问题3：WSL 中无法访问 localhost:6006

确保 Windows 的端口转发正常。如果不行，可以：

1. 在 Windows PowerShell 中运行：
```powershell
netsh interface portproxy add v4tov4 listenport=6006 listenaddress=0.0.0.0 connectport=6006 connectaddress=localhost
```

2. 或者直接在 WSL 的 IP 地址访问（需要先查看 WSL IP）：
```bash
# 查看 WSL IP
hostname -I
# 然后在浏览器访问：http://<WSL_IP>:6006
```

---

## 💡 实用技巧

### 1. 对比多次训练

```bash
# 同时查看多个训练运行（在同一个 TensorBoard 中）
tensorboard --logdir ./runs --port 6006
```

TensorBoard 会自动显示所有子目录的指标，可以切换查看。

### 2. 实时监控训练

在训练过程中，可以同时运行 TensorBoard，它会自动更新：

```bash
# 终端1：训练
python3 step_08_train.py

# 终端2：查看 TensorBoard
tensorboard --logdir ./runs --port 6006
```

### 3. 导出数据

TensorBoard 的数据可以导出为 CSV（需要安装插件）：

```bash
pip install tensorboard-plugin-profile
```

---

## 🎯 快速命令总结

```bash
# 启动 TensorBoard（查看所有运行）
tensorboard --logdir ./runs --port 6006

# 查看最新的一次训练
LATEST_RUN=$(ls -td ./runs/ppo_pendulum_* | head -1)
tensorboard --logdir $LATEST_RUN --port 6006

# 后台运行
nohup tensorboard --logdir ./runs --port 6006 > tensorboard.log 2>&1 &

# 停止 TensorBoard
pkill -f tensorboard
```

---

**启动后，在浏览器访问 `http://localhost:6006` 即可查看训练曲线！** 🎉

