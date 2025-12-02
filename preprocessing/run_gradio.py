#!/usr/bin/env python3
"""
启动gradio界面的脚本
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gradio_interface import create_interface

if __name__ == "__main__":
    print("正在启动视频帧处理工具的Web界面...")
    print("请确保已安装所有依赖包：pip install -r requirements.txt")
    print("请确保系统中已安装 COLMAP 和 FFmpeg")
    print("界面将在 http://localhost:7860 启动")
    print("按 Ctrl+C 停止服务器")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True  # 自动打开浏览器
    ) 