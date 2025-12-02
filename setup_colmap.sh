#!/bin/bash

# colmap安装配置脚本（用于全景相机位姿估计）

set -e

echo "==== colmap安装开始 ===="
echo "注意：脚本中会使用 sudo 命令，请提前输入密码："
sudo -v

echo "==== 8. 安装 FFmpeg ===="
sudo apt install ffmpeg

echo "==== 9. 编译安装 SphereSfM ===="
# 检查 submodules 目录是否存在
if [ ! -d "submodules" ]; then
    echo "未找到 submodules 目录，请确保在项目根目录运行此脚本。"
    exit 1
fi

# 检查 SphereSfM 目录是否存在
if [ ! -d "submodules/SphereSfM" ]; then
    echo "未找到 submodules/SphereSfM 目录，请先初始化 git submodules。"
    echo "运行: git submodule update --init --recursive"
    exit 1
fi

# 安装额外依赖
sudo apt install -y libboost-filesystem-dev libboost-system-dev
sudo apt install -y libflann-dev

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 进入 SphereSfM 目录
cd submodules/SphereSfM

echo "==== 9.1. 清理之前的编译文件 ===="
# 删除 build 目录（如果存在）
if [ -d "build" ]; then
    echo "删除旧的 build 目录..."
    rm -rf build
fi

# 删除 CMakeCache.txt 和其他 CMake 缓存文件（如果存在）
if [ -f "CMakeCache.txt" ]; then
    echo "删除 CMakeCache.txt..."
    rm -f CMakeCache.txt
fi

# 删除 CMakeFiles 目录（如果存在）
if [ -d "CMakeFiles" ]; then
    echo "删除 CMakeFiles 目录..."
    rm -rf CMakeFiles
fi

# 删除其他可能的 CMake 生成文件
find . -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "cmake_install.cmake" -delete 2>/dev/null || true
find . -name "Makefile" -delete 2>/dev/null || true

echo "清理完成，开始全新编译..."

# 创建并进入 build 目录
mkdir -p build
cd build

# 检测GPU架构的更可靠方法
echo "检测GPU架构..."
ARCH=""

# 方法1：尝试使用nvidia-smi检测实际GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "检测到GPU: $GPU_NAME"
    
    # 根据GPU名称推断架构
    if [[ "$GPU_NAME" == *"5090"* ]] || [[ "$GPU_NAME" == *"5080"* ]] || [[ "$GPU_NAME" == *"5070"* ]] || [[ "$GPU_NAME" == *"5060"* ]] || [[ "$GPU_NAME" == *"5050"* ]] || [[ "$GPU_NAME" == *"RTX 50"* ]]; then
        ARCH="12.0"  # RTX 50系列消费级显卡，Blackwell架构
    elif [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"4080"* ]] || [[ "$GPU_NAME" == *"RTX 40"* ]]; then
        ARCH="89"  # RTX 40系列，Ada Lovelace架构
    elif [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"3080"* ]] || [[ "$GPU_NAME" == *"RTX 30"* ]]; then
        ARCH="86"  # RTX 30系列，Ampere架构
    elif [[ "$GPU_NAME" == *"2080"* ]] || [[ "$GPU_NAME" == *"RTX 20"* ]]; then
        ARCH="75"  # RTX 20系列，Turing架构
    elif [[ "$GPU_NAME" == *"1080"* ]] || [[ "$GPU_NAME" == *"GTX 10"* ]]; then
        ARCH="61"  # GTX 10系列，Pascal架构
    elif [[ "$GPU_NAME" == *"A4000"* ]] || [[ "$GPU_NAME" == *"A5000"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
        ARCH="86"  # A系列专业显卡，Ampere架构
    elif [[ "$GPU_NAME" == *"V100"* ]]; then
        ARCH="70"  # V100，Volta架构
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        ARCH="80"  # A100，Ampere架构
    elif [[ "$GPU_NAME" == *"H100"* ]]; then
        ARCH="90"  # H100，Hopper架构
    elif [[ "$GPU_NAME" == *"B100"* ]] || [[ "$GPU_NAME" == *"B200"* ]]; then
        ARCH="10.0"  # B系列数据中心显卡，Blackwell架构
    elif [[ "$GPU_NAME" == *"Quadro"* ]]; then
        # Quadro系列显卡，需要根据具体型号判断
        if [[ "$GPU_NAME" == *"RTX"* ]]; then
            ARCH="75"  # Quadro RTX系列，一般是Turing架构
        else
            ARCH="61"  # 老的Quadro系列
        fi
    fi
fi

# 方法2：如果nvidia-smi检测失败，尝试从nvcc解析支持的架构
if [ -z "$ARCH" ]; then
    echo "尝试从nvcc解析架构..."
    # 首先尝试找到最高的数字架构（如sm_120）
    ARCH_NUM=$(nvcc --help | grep -o "sm_[0-9][0-9]*" | sed 's/sm_//' | sort -n | tail -n1)
    if [ -n "$ARCH_NUM" ]; then
        # 将数字架构转换为compute capability格式
        if [ "$ARCH_NUM" -ge 120 ]; then
            ARCH="12.0"
        elif [ "$ARCH_NUM" -ge 100 ]; then
            ARCH="10.0"
        else
            ARCH="$ARCH_NUM"
        fi
    fi
fi

# 方法3：如果以上都失败，使用默认值
if [ -z "$ARCH" ]; then
    echo "无法自动检测GPU架构，使用默认架构 native"
    ARCH="native"
fi

echo "使用的CUDA架构: $ARCH"

# 配置 CMake
echo "配置 CMake..."
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=$ARCH

# 编译
echo "编译 SphereSfM..."
ninja

# 安装
echo "安装 SphereSfM..."
sudo ninja install

# 返回项目根目录
cd ../../..

# 测试colmap安装
colmap gui