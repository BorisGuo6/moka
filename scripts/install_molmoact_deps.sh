#!/bin/bash
# MolmoAct 依赖安装脚本

echo "🚀 安装 MolmoAct 依赖..."

# 检查是否在 conda 环境中
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "⚠️ 建议在 conda 环境中运行此脚本"
    echo "请先运行: conda activate moka"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装必要的 Python 包
echo "📦 安装 Python 包..."
pip install transformers>=4.44.0
pip install torch>=2.0.0
pip install pillow
pip install requests
pip install termcolor
pip install einops
pip install accelerate

# 检查 CUDA 支持
echo "🔍 检查 CUDA 支持..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"

echo "✅ 依赖安装完成!"
echo ""
echo "使用方法:"
echo "  python scripts/quick_test_molmoact.py    # 快速测试"
echo "  python scripts/test_molmoact.py          # 完整测试"
