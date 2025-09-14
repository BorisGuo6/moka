#!/bin/bash

# Grounded-SAM 完整环境配置脚本
# 自动检测CUDA版本并配置所有必要的环境变量

echo "🚀 Grounded-SAM Environment Setup"
echo "=================================="

# 设置基本环境变量
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

# CUDA自动检测函数
detect_cuda() {
    local cuda_path=""
    
    # 方法1: 检查nvcc命令
    if command -v nvcc &> /dev/null; then
        cuda_path=$(dirname $(dirname $(which nvcc)))
        echo "✅ Found CUDA via nvcc: $cuda_path"
    else
        # 方法2: 检查常见安装路径
        local common_paths=(
            "/usr/local/cuda"
            "/usr/local/cuda-12.6" "/usr/local/cuda-12.5" "/usr/local/cuda-12.4" "/usr/local/cuda-12.3"
            "/usr/local/cuda-12.2" "/usr/local/cuda-12.1" "/usr/local/cuda-12.0"
            "/usr/local/cuda-11.8" "/usr/local/cuda-11.7" "/usr/local/cuda-11.6"
            "/opt/cuda" "/opt/cuda-12.6" "/opt/cuda-12.5"
        )
        
        for path in "${common_paths[@]}"; do
            if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
                cuda_path="$path"
                echo "✅ Found CUDA at: $cuda_path"
                break
            fi
        done
        
        # 方法3: 检查环境变量
        if [ -z "$cuda_path" ] && [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
            cuda_path="$CUDA_HOME"
            echo "✅ Using existing CUDA_HOME: $cuda_path"
        fi
    fi
    
    if [ -z "$cuda_path" ]; then
        echo "❌ CUDA not found!"
        echo "Please install CUDA or set CUDA_HOME manually."
        echo "Common paths: /usr/local/cuda, /usr/local/cuda-12.x"
        return 1
    fi
    
    # 验证CUDA安装
    if [ ! -f "$cuda_path/bin/nvcc" ]; then
        echo "❌ CUDA installation incomplete at $cuda_path"
        return 1
    fi
    
    echo "$cuda_path"
    return 0
}

# 检测CUDA版本
get_cuda_version() {
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/'
    else
        echo "unknown"
    fi
}

# 主检测逻辑
echo "🔍 Auto-detecting CUDA installation..."

CUDA_HOME=$(detect_cuda)
if [ $? -ne 0 ]; then
    exit 1
fi

CUDA_VERSION=$(get_cuda_version)
echo "📋 Detected CUDA version: $CUDA_VERSION"

# 设置环境变量
export CUDA_HOME="$CUDA_HOME"
export CUDA_VISIBLE_DEVICES=0

# 添加CUDA到PATH
if [[ ":$PATH:" != *":$CUDA_HOME/bin:"* ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# 添加CUDA库路径
if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA_HOME/lib64:"* ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# 显示配置结果
echo ""
echo "📊 Environment Configuration:"
echo "  AM_I_DOCKER=$AM_I_DOCKER"
echo "  BUILD_WITH_CUDA=$BUILD_WITH_CUDA"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  CUDA_VERSION=$CUDA_VERSION"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  PATH includes: $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"

# 检查GPU状态
echo ""
echo "🖥️  GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
else
    echo "nvidia-smi not found. GPU status unknown."
fi

# 更新bashrc选项
echo ""
echo "💾 Permanent Configuration:"
echo "To make these changes permanent, run:"
echo "  echo 'export AM_I_DOCKER=False' >> ~/.bashrc"
echo "  echo 'export BUILD_WITH_CUDA=True' >> ~/.bashrc"
echo "  echo 'export CUDA_HOME=$CUDA_HOME' >> ~/.bashrc"
echo "  echo 'export PATH=\$CUDA_HOME/bin:\$PATH' >> ~/.bashrc"
echo "  echo 'export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo "  echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc"

# 自动更新bashrc选项
read -p "🤔 Do you want to automatically update ~/.bashrc? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Updating ~/.bashrc..."
    
    # 删除旧的Grounded-SAM配置
    if grep -q "Grounded-SAM Environment Variables" ~/.bashrc; then
        sed -i '/# Grounded-SAM Environment Variables/,/^$/d' ~/.bashrc
    fi
    
    # 添加新配置
    echo "" >> ~/.bashrc
    echo "# Grounded-SAM Environment Variables" >> ~/.bashrc
    echo "export AM_I_DOCKER=False" >> ~/.bashrc
    echo "export BUILD_WITH_CUDA=True" >> ~/.bashrc
    echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    
    echo "✅ ~/.bashrc updated successfully!"
    echo "To apply changes, run: source ~/.bashrc"
fi

echo ""
echo "🎉 Grounded-SAM environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: cd grounded-sam && pip install -r requirements.txt"
echo "2. Download model weights from the Grounded-SAM repository"
echo "3. Run demos: python grounded_sam_demo.py"
