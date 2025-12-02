# 视频帧处理工具 - Gradio界面

这个工具提供了一个友好的Web界面来处理视频，提取帧并进行相机标定、去畸变和下采样处理。

## 功能特点

- 🎥 从视频中提取帧
- 📷 使用COLMAP进行相机标定
- 🔧 图像去畸变处理
- 📏 多尺度下采样
- 🖥️ 用户友好的Web界面
- 📊 实时进度显示

## 安装依赖

### 1. Python依赖
```bash
pip install -r requirements.txt
```

### 2. 系统依赖
确保系统中已安装以下软件：

- **COLMAP**: 用于相机标定
  ```bash
  # Ubuntu/Debian
  sudo apt-get install colmap
  
  # macOS
  brew install colmap
  ```

- **FFmpeg**: 用于视频处理
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg
  ```

## 使用方法

### 1. 启动界面
```bash
python run_gradio.py
```

或者直接运行：
```bash
python gradio_interface.py
```

### 2. 访问界面
启动后，会自动打开浏览器访问 http://localhost:7860

### 3. 使用步骤
1. 在"处理设置"选项卡中：
   - 输入视频文件夹路径
   - 输入输出文件夹路径
   - 设置总帧数（默认100）
   - 选择匹配方法（推荐vocab_tree）
   - 选择是否细化内参（推荐开启）

2. 检查系统状态：
   - 确保COLMAP和FFmpeg都显示可用状态
   - 如果显示不可用，请安装相应软件

3. 点击"开始处理"按钮开始处理

4. 观察进度显示，等待处理完成

## 支持的视频格式

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

## 处理步骤

1. **提取视频帧**: 从视频中按时间均匀提取帧
2. **相机标定**: 使用COLMAP进行特征提取和匹配
3. **去畸变**: 修正相机畸变
4. **下采样**: 生成多个尺度的图像（2x, 4x, 8x, 16x）

## 输出结果

处理完成后，输出文件夹将包含：

```
output_folder/
├── images/                    # 提取的原始帧
├── sparse/                    # COLMAP稀疏重建结果
├── undistorted/              # 去畸变后的图像和相机参数
├── images_2/                 # 2倍下采样图像
├── images_4/                 # 4倍下采样图像
├── images_8/                 # 8倍下采样图像
└── images_16/                # 16倍下采样图像
```

## 参数说明

- **总帧数**: 从所有视频中提取的总帧数，会根据视频时长自动分配
- **匹配方法**:
  - `vocab_tree`: 使用词汇树匹配（推荐，速度快）
  - `exhaustive`: 穷举匹配（精度高，速度慢）
  - `sequential`: 顺序匹配（适合序列图像）
- **细化内参**: 是否在束集调整中优化相机内参

## 故障排除

### 1. COLMAP未找到
- 确保COLMAP已安装并添加到系统PATH
- 尝试在终端中运行 `colmap -h` 验证

### 2. FFmpeg未找到
- 确保FFmpeg已安装并添加到系统PATH
- 尝试在终端中运行 `ffmpeg -version` 验证

### 3. 处理失败
- 检查输入文件夹是否包含支持的视频格式
- 确保输出文件夹有写权限
- 查看界面中的错误信息

### 4. 内存不足
- 减少总帧数设置
- 使用sequential匹配方法而不是vocab_tree

## 注意事项

- 处理时间取决于视频数量、帧数和匹配方法
- vocab_tree匹配在首次使用时会下载词汇树文件（约100MB）
- 确保有足够的磁盘空间存储输出结果
- 处理过程中请勿关闭浏览器窗口

## 许可证

请参考项目主许可证文件。 