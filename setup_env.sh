#!/bin/bash

# 一键环境配置脚本（适用于 WSL Ubuntu）

set -e

echo "==== 环境配置开始 ===="
echo "注意：脚本中会使用 sudo 命令，请提前输入密码："
sudo -v

echo "==== 1. 安装 Miniconda ===="
if ! command -v conda &> /dev/null; then
    # 检查 requirements 目录中是否有 Miniconda 安装包
    if [ -f "requirements/Miniconda3-latest-Linux-x86_64.sh" ]; then
        echo "使用 requirements 目录中的 Miniconda 安装包..."
        bash requirements/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    else
        echo "requirements 目录中未找到 Miniconda 安装包，正在下载..."
        wget -O requirements/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash requirements/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    fi
    
    # 手动初始化 conda
    echo "初始化 conda..."
    $HOME/miniconda3/bin/conda init bash
    
    # 重新加载环境变量
    source ~/.bashrc
    
    echo "Conda 安装完成"
else
    echo "已检测到 conda，无需重复安装。"
fi

echo "==== 2. 安装系统依赖（包括 COLMAP 编译依赖） ===="
sudo apt update

# 设置非交互式安装，使用默认选项（通常是 No）
export DEBIAN_FRONTEND=noninteractive

sudo apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libmkl-full-dev

echo "==== 3. 创建并激活 Conda 环境 ===="
# 确保 conda 命令可用
source ~/miniconda3/etc/profile.d/conda.sh
if ! command -v conda &> /dev/null; then
    source ~/.bashrc
fi

if conda info --envs | grep -q "^gspl"; then
    echo "Conda 环境 gspl 已存在。"
else
    conda create -yn gspl python=3.9 pip
fi

conda activate gspl

echo "==== 4. 安装 PyTorch 及 CUDA 对应版本 ===="
pip install -r requirements/pyt271_cu128.txt

echo "==== 5. 安装 CUDA Toolkit ===="
if [ ! -f "requirements/cuda_12.8.0_570.86.10_linux.run" ]; then
    echo "未找到 CUDA 安装包，请先下载到 requirements 目录。"
    exit 1
fi
sudo sh requirements/cuda_12.8.0_570.86.10_linux.run --silent --toolkit

echo "==== 6. 重新加载环境变量（CUDA 安装后） ===="
source ~/.bashrc
# 重新激活 conda 环境
conda activate gspl

echo "==== 7. 安装其他 Python 依赖 ===="
pip install -r requirements.txt
pip install -r submodules/PanDA/requirements.txt

echo "==== 8. 最终环境变量重新加载 ===="
source ~/.bashrc
# 再次重新激活 conda 环境
conda activate gspl

echo "==== 环境配置完成！===="
echo ""
echo "重要提示："
echo "1. 请重新打开终端或运行: source ~/.bashrc"
echo "2. 验证 conda 环境: conda info"
echo "3. 激活环境: conda activate gspl"