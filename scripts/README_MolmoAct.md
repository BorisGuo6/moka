# MolmoAct 测试脚本

这个目录包含了用于测试 AllenAI MolmoAct-7B-D-0812 模型的脚本。

## 文件说明

- `test_molmoact.py` - 完整的测试脚本，包含详细的输出和错误处理
- `quick_test_molmoact.py` - 快速测试脚本，用于验证模型是否正常工作
- `install_molmoact_deps.sh` - 依赖安装脚本
- `README_MolmoAct.md` - 本说明文件

## 快速开始

### 1. 安装依赖

```bash
# 激活 conda 环境
conda activate moka

# 运行安装脚本
bash scripts/install_molmoact_deps.sh
```

或者手动安装：

```bash
pip install transformers>=4.44.0 torch>=2.0.0 pillow requests termcolor
```

### 2. 运行测试

#### 快速测试（推荐首次使用）
```bash
python scripts/quick_test_molmoact.py
```

#### 完整测试
```bash
python scripts/test_molmoact.py
```

## 模型说明

MolmoAct-7B-D-0812 是一个多模态模型，能够：

- 📸 **图像理解**: 从多个相机视角理解场景
- 🧠 **推理**: 进行深度感知和轨迹分析
- 🎮 **动作生成**: 输出 7-DoF 机器人动作（位置 + 旋转 + 夹爪状态）

### 输入格式
- **图像**: 多个相机视角的 RGB 图像
- **文本**: 自然语言任务描述（如 "close the box"）

### 输出格式
- **深度感知**: 深度图标记
- **视觉轨迹**: 末端执行器轨迹点
- **动作**: 7维动作向量 `[x, y, z, rx, ry, rz, gripper]`

## 示例输出

```
📝 生成的文本:
--------------------------------------------------------------------------------
The depth map of the first image shows a box on a table surface with clear depth 
information. The trajectory of the end effector should approach the box from above 
and close it by moving the lid down. Based on this analysis, the action that the 
robot should take is to move to the box position and close it.

🎮 解析的动作: [[0.0732, 0.0823, -0.0278, 0.1593, -0.0969, 0.0439, 0.9961]]
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 使用 CPU 模式
   export CUDA_VISIBLE_DEVICES=""
   python scripts/quick_test_molmoact.py
   ```

2. **模型下载失败**
   ```bash
   # 设置 Hugging Face 缓存目录
   export HF_HOME=/path/to/large/disk
   python scripts/quick_test_molmoact.py
   ```

3. **依赖版本冲突**
   ```bash
   # 创建新的 conda 环境
   conda create -n molmoact python=3.10
   conda activate molmoact
   bash scripts/install_molmoact_deps.sh
   ```

### 性能优化

- **GPU 内存**: 模型需要约 14GB GPU 内存
- **生成速度**: 在 RTX 4090 上约需 10-30 秒
- **批处理**: 可以同时处理多个任务

## 技术细节

### 模型架构
- **基础模型**: 7B 参数的多模态 Transformer
- **输入处理**: 图像编码器 + 文本编码器
- **输出解析**: 专门的解析器提取深度、轨迹和动作

### 推理流程
1. 图像预处理和编码
2. 文本提示构建
3. 多模态融合
4. 逐步推理生成
5. 输出解析和反归一化

## 参考资料

- [MolmoAct 论文](https://arxiv.org/abs/2408.08112)
- [Hugging Face 模型页面](https://huggingface.co/allenai/MolmoAct-7B-D-0812)
- [AllenAI 官网](https://allenai.org/)
