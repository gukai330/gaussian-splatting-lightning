import gradio as gr
import os
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Literal
from packaging.version import Version
import argparse
from PIL import Image
import numpy as np
import sys
import threading
import time
import shutil
import tkinter as tk
from tkinter import filedialog

# 导入原有的处理函数
from process_images import (
    extract_frames_ffmpeg,
    run_colmap,
    run_undistort,
    downscale_images,
    get_colmap_version,
    get_vocab_tree,
    run_command
)

# 导入深度mask生成函数
from generate_depth_mask import (
    process_image_directory,
    generate_masks_with_downsampling
)

# 全局变量用于跟踪进度
progress_info = {"step": "", "progress": 0, "total": 0, "message": ""}
processing_active = False

def update_progress(step: str, progress: int = 0, total: int = 100, message: str = ""):
    """更新进度信息"""
    global progress_info
    progress_info.update({
        "step": step,
        "progress": progress,
        "total": total,
        "message": message
    })

def get_image_resolution(image_dir):
    """获取图像分辨率"""
    import glob
    from PIL import Image
    
    # 查找第一个图像文件
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files = glob.glob(os.path.join(image_dir, ext))
        if image_files:
            img = Image.open(image_files[0])
            return img.width, img.height
    
    raise ValueError(f"在目录 {image_dir} 中没有找到图像文件")

def estimate_depth(input_folder, image_dir="images"):
    """估计数据集深度"""
    import subprocess
    import sys
    
    utils_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    estimate_script = os.path.join(utils_dir, "utils", "estimate_dataset_depths.py")
    image_path = os.path.join(input_folder, image_dir)
    
    if not os.path.exists(image_path):
        raise ValueError(f"图像目录不存在: {image_path}")
    
    # 运行深度估计脚本
    cmd = [sys.executable, estimate_script, input_folder, "--image_dir", image_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"深度估计失败: {result.stderr}")
    
    return f"深度估计完成，结果保存在: {os.path.join(input_folder, 'estimated_depths')}"

def estimate_depth_panda(input_folder, image_dir="images"):
    """使用PanDA进行全景图像深度估计"""
    import subprocess
    import sys
    
    # PanDA相关路径
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    panda_dir = os.path.join(project_dir, "submodules", "PanDA")
    run_image_script = os.path.join(panda_dir, "run_image.py")
    config_file = os.path.join(panda_dir, "config", "inference", "panda_large.yaml")
    
    image_path = os.path.join(input_folder, image_dir)
    output_dir = os.path.join(input_folder, "estimated_depths")
    
    if not os.path.exists(image_path):
        raise ValueError(f"图像目录不存在: {image_path}")
    
    if not os.path.exists(config_file):
        raise ValueError(f"找不到PanDA配置文件: {config_file}")
    
    # 运行PanDA深度估计
    cmd = [
        sys.executable, run_image_script,
        "--config", config_file,
        "--img-path", image_path,
        "--outdir", output_dir,
        "--resize"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=panda_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"PanDA深度估计失败: {result.stderr}")
    
    # 运行深度缩放估计
    utils_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scale_script = os.path.join(utils_dir, "utils", "get_depth_scales_pano.py")
    
    cmd_scale = [sys.executable, scale_script, input_folder]
    result_scale = subprocess.run(cmd_scale, capture_output=True, text=True)
    
    if result_scale.returncode != 0:
        raise RuntimeError(f"深度缩放估计失败: {result_scale.stderr}")
    
    return f"PanDA深度估计完成，结果保存在: {output_dir}"

def extract_frames_with_progress(input_folder, output_folder, total_frames=10, progress_callback=None):
    """带进度回调的视频帧提取函数"""
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    durations = []
    
    # 获取所有视频的时长
    for i, file in enumerate(video_files):
        if progress_callback:
            progress_callback(f"分析视频 {i+1}/{len(video_files)}: {file}")
        path = os.path.join(input_folder, file)
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{path}\""
        try:
            duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
            durations.append((file, duration))
        except:
            durations.append((file, 0))
    
    total_duration = sum(d for _, d in durations)
    if total_duration == 0:
        raise ValueError("无法读取视频时长")
    
    # 计算每个视频分配的帧数
    alloc = {f: max(1, int(total_frames * (d / total_duration))) for f, d in durations}
    
    # 计算总的帧数用于进度计算
    total_frames_to_extract = sum(alloc.values())
    extracted_frames = 0
    
    # 提取帧
    for file, dur in durations:
        if dur == 0: 
            continue
        
        video_path = os.path.join(input_folder, file)
        prefix = os.path.join(output_folder, Path(file).stem)
        frames = alloc[file]
        timestamps = np.linspace(0, dur, frames)
        
        for i, t in enumerate(timestamps):
            if progress_callback:
                progress_callback(f"提取帧 {extracted_frames + 1}/{total_frames_to_extract}: {file} 帧{i+1}/{frames}")
            
            outname = f"{prefix}_frame{i}.png"
            cmd = f"ffmpeg -ss {t:.2f} -i \"{video_path}\" -frames:v 1 -pix_fmt rgb24 -y \"{outname}\""
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            extracted_frames += 1
    
    return extracted_frames

def run_colmap_with_camera_model(
    image_dir: Path,
    colmap_dir: Path,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    refine_intrinsics: bool = True,
    camera_model: str = "OPENCV",
    max_num_matches: int = 32768,
    colmap_cmd: str = "colmap",
):
    """带相机模型选择的COLMAP函数"""
    db = colmap_dir / "database.db"
    db.unlink(missing_ok=True)
    version = get_colmap_version(colmap_cmd)

    # Feature Extraction
    if camera_model == "SPHERE":
        # Sphere模式：获取图像分辨率并计算相机参数
        width, height = get_image_resolution(str(image_dir))
        camera_params = f"1,{width//2},{height//2}"
        
        run_command(f"{colmap_cmd} feature_extractor "
                    f"--database_path {db} "
                    f"--image_path {image_dir} "
                    f"--ImageReader.camera_model SPHERE "
                    f"--ImageReader.camera_params \"{camera_params}\" "
                    f"--ImageReader.single_camera 1 "
                    f"--SiftExtraction.use_gpu 1", verbose=True)
    else:
        # 普通模式
        run_command(f"{colmap_cmd} feature_extractor "
                    f"--database_path {db} "
                    f"--image_path {image_dir} "
                    f"--ImageReader.single_camera 1 "
                    f"--SiftExtraction.use_gpu 1 "
                    f"--ImageReader.camera_model {camera_model}", verbose=True)

    # Feature Matching - sphere模式不支持spatial matcher
    if camera_model == "SPHERE" and matching_method not in ["vocab_tree", "exhaustive", "sequential"]:
        matching_method = "vocab_tree"  # 默认使用vocab_tree
    
    match_cmd = f"{colmap_cmd} {matching_method}_matcher --database_path {db} --SiftMatching.use_gpu 1 --SiftMatching.max_num_matches {max_num_matches}"
    if matching_method == "vocab_tree":
        vocab = get_vocab_tree()
        match_cmd += f" --VocabTreeMatching.vocab_tree_path \"{vocab}\""
    run_command(match_cmd, True)

    # Mapping
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    if camera_model == "SPHERE":
        # Sphere模式：固定相机参数，不进行优化
        mapper_cmd = (f"{colmap_cmd} mapper --database_path {db} "
                      f"--image_path {image_dir} "
                      f"--output_path {sparse_dir} "
                      f"--Mapper.ba_refine_focal_length 0 "
                      f"--Mapper.ba_refine_principal_point 0 "
                      f"--Mapper.ba_refine_extra_params 0 "
                      f"--Mapper.sphere_camera 1")
    else:
        # 普通模式
        mapper_cmd = (f"{colmap_cmd} mapper --database_path {db} "
                      f"--image_path {image_dir} "
                      f"--output_path {sparse_dir}")
    
    if version >= Version("3.7"):
        mapper_cmd += " --Mapper.ba_global_function_tolerance=1e-6"
    run_command(mapper_cmd, verbose=True)

    # Bundle adjustment - sphere模式不进行内参优化
    if refine_intrinsics and camera_model != "SPHERE":
        ba_cmd = f"{colmap_cmd} bundle_adjuster --input_path {sparse_dir}/0 --output_path {sparse_dir}/0 --BundleAdjustment.refine_principal_point 1"
        run_command(ba_cmd, verbose=True)

def check_step_prerequisites(step_name: str, input_folder: str) -> str:
    """检查步骤的前置条件"""
    if not input_folder:
        return "错误：请选择输入文件夹"
    
    # 输出文件夹设置为输入文件夹
    output_path = Path(input_folder)
    
    if step_name == "extract_frames":
        if not os.path.exists(input_folder):
            return "错误：输入文件夹不存在"
        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            return "错误：输入文件夹中没有找到视频文件"
    
    elif step_name == "colmap":
        images_path = output_path / "images"
        if not images_path.exists():
            return "错误：请先提取视频帧"
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return "错误：images文件夹中没有找到图像文件"
    
    elif step_name == "undistort":
        sparse_path = output_path / "sparse"
        if not sparse_path.exists():
            return "错误：请先完成相机标定"
        # 检查sparse文件夹中是否有重建结果
        has_reconstruction = False
        for sub_dir in sparse_path.iterdir():
            if sub_dir.is_dir() and (sub_dir / "cameras.bin").exists():
                has_reconstruction = True
                break
        if not has_reconstruction:
            return "错误：没有找到有效的相机标定结果"
    
    elif step_name == "downscale":
        # 检查可用于下采样的图像目录：优先undistorted，然后原始images
        undistorted_path = output_path / "undistorted" / "images"
        images_path = output_path / "images"
        
        if undistorted_path.exists():
            image_files = [f for f in os.listdir(undistorted_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                return "错误：undistorted/images文件夹中没有找到图像文件"
        elif images_path.exists():
            image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                return "错误：images文件夹中没有找到图像文件"
        else:
            return "错误：请先提取视频帧或完成相机标定"
    
    elif step_name == "estimate_depth":
        # 检查是否有图像文件用于深度估计
        # 按优先级检查：下采样图像 > undistorted图像 > 原始图像
        image_dir_found = False
        
        # 检查所有可能的图像目录（包括undistorted和当前目录）
        check_paths = [
            output_path / "undistorted" / "images_2",  # 去畸变后2倍下采样
            output_path / "undistorted" / "images_4",  # 去畸变后4倍下采样
            output_path / "images_2",                  # 当前目录2倍下采样
            output_path / "images_4",                  # 当前目录4倍下采样
            output_path / "undistorted" / "images",    # 去畸变图像
            output_path / "images"                     # 原始图像
        ]
        
        for path in check_paths:
            if path.exists():
                image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    image_dir_found = True
                    break
        
        if not image_dir_found:
            return "错误：请先提取视频帧或完成相机标定"
    
    elif step_name == "downscale_masks":
        # 检查是否有 masks 目录可用于下采样
        # 按优先级检查：undistorted/masks > masks
        undistorted_masks = output_path / "undistorted/masks"
        masks_path = output_path / "masks"
        
        if undistorted_masks.exists():
            mask_files = [f for f in os.listdir(undistorted_masks) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not mask_files:
                return "错误：undistorted/masks文件夹中没有找到mask文件"
        elif masks_path.exists():
            mask_files = [f for f in os.listdir(masks_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not mask_files:
                return "错误：masks文件夹中没有找到mask文件"
        else:
            return "错误：请先生成masks或完成相机标定"
    
    elif step_name == "generate_mask":
        # 检查是否存在深度估计结果
        depth_dir = output_path / "estimated_depths"
        if not depth_dir.exists():
            return "错误：请先完成深度估计"
        
        # 检查深度文件是否存在
        depth_files = list(depth_dir.glob("*.npy"))
        if not depth_files:
            return "错误：深度估计目录中没有找到深度文件"
        
        # 检查是否有对应的图像文件
        image_dir_found = False
        check_paths = [
            output_path / "undistorted" / "images",    # 去畸变图像
            output_path / "images"                     # 原始图像
        ]
        
        for path in check_paths:
            if path.exists():
                image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    image_dir_found = True
                    break
        
        if not image_dir_found:
            return "错误：没有找到可用于生成mask的图像文件"
    
    return ""

def step_extract_frames(input_folder: str, output_folder: str, total_frames: int, progress=gr.Progress()):
    """步骤1：提取视频帧"""
    global processing_active
    
    error_msg = check_step_prerequisites("extract_frames", input_folder)
    if error_msg:
        return error_msg
    
    processing_active = True
    
    try:
        output_path = Path(output_folder)
        images_path = output_path / "images"
        
        def progress_callback(message):
            update_progress("提取视频帧", 50, 100, message)
        
        progress(0.1, desc="正在提取视频帧...")
        update_progress("提取视频帧", 10, 100, "开始提取视频帧")
        
        extracted_count = extract_frames_with_progress(input_folder, images_path, total_frames, progress_callback)
        
        progress(1.0, desc="提取完成！")
        update_progress("提取视频帧", 100, 100, "视频帧提取完成")
        
        processing_active = False
        return f"✅ 视频帧提取完成！共提取了 {extracted_count} 帧到：{images_path}"
        
    except Exception as e:
        processing_active = False
        return f"❌ 提取视频帧失败：{str(e)}"

def step_colmap(input_folder: str, output_folder: str, matching_method: Literal["vocab_tree", "exhaustive", "sequential"], refine_intrinsics: bool, camera_model: str, progress=gr.Progress()):
    """步骤2：相机标定"""
    global processing_active
    
    error_msg = check_step_prerequisites("colmap", input_folder)
    if error_msg:
        return error_msg
    
    processing_active = True
    
    try:
        output_path = Path(output_folder)
        images_path = output_path / "images"
        
        progress(0.1, desc="正在进行相机标定...")
        update_progress("相机标定", 10, 100, f"开始COLMAP相机标定 (相机模型: {camera_model})")
        
        run_colmap_with_camera_model(
            image_dir=images_path,
            colmap_dir=output_path,
            matching_method=matching_method,
            refine_intrinsics=refine_intrinsics,
            camera_model=camera_model,
        )
        
        progress(1.0, desc="相机标定完成！")
        update_progress("相机标定", 100, 100, "COLMAP相机标定完成")
        
        processing_active = False
        return f"✅ 相机标定完成！使用相机模型：{camera_model}，结果保存在：{output_path / 'sparse'}"
        
    except Exception as e:
        processing_active = False
        return f"❌ 相机标定失败：{str(e)}"

def step_undistort(input_folder: str, output_folder: str, progress=gr.Progress()):
    """步骤3：去畸变"""
    global processing_active
    
    error_msg = check_step_prerequisites("undistort", input_folder)
    if error_msg:
        return error_msg
    
    processing_active = True
    
    try:
        output_path = Path(output_folder)
        
        progress(0.1, desc="正在去畸变...")
        update_progress("去畸变", 10, 100, "开始图像去畸变")
        
        run_undistort(output_path)
        
        progress(1.0, desc="去畸变完成！")
        update_progress("去畸变", 100, 100, "图像去畸变完成")
        
        processing_active = False
        return f"✅ 去畸变完成！结果保存在：{output_path / 'undistorted'}"
        
    except Exception as e:
        processing_active = False
        return f"❌ 去畸变失败：{str(e)}"


def downscale_masks_dir(masks_root: Path, scales = [2, 4]):
    """将 masks 目录按给定倍数下采样到同级的 masks_{scale} 目录（最近邻）。"""
    image_exts = [".png", ".jpg", ".jpeg"]
    for scale in scales:
        out_dir = masks_root.parent / f"masks_{scale}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for mask_path in masks_root.iterdir():
            if not mask_path.is_file():
                continue
            if mask_path.suffix.lower() not in image_exts:
                continue
            try:
                with Image.open(mask_path) as im:
                    new_w = max(1, im.width // scale)
                    new_h = max(1, im.height // scale)
                    resized = im.resize((new_w, new_h), Image.NEAREST)
                    resized.save(out_dir / mask_path.name)
            except Exception:
                # 忽略单个文件错误
                continue

def step_downscale(input_folder: str, output_folder: str, progress=gr.Progress()):
    """步骤4：下采样"""
    global processing_active
    
    error_msg = check_step_prerequisites("downscale", input_folder)
    if error_msg:
        return error_msg
    
    processing_active = True
    
    try:
        output_path = Path(output_folder)
        undistorted_images = output_path / "undistorted/images"
        original_images = output_path / "images"

        progress(0.1, desc="正在下采样...")
        update_progress("下采样", 10, 100, "开始多尺度下采样")

        # 选择下采样源目录（与批量处理逻辑保持一致）
        if undistorted_images.exists():
            source_images = undistorted_images
            base_dir = output_path / "undistorted"
            update_progress("下采样", 20, 100, "从去畸变图像进行下采样")
        elif original_images.exists():
            source_images = original_images
            base_dir = output_path
            update_progress("下采样", 20, 100, "从原始图像进行下采样")
        else:
            processing_active = False
            return "错误：找不到可下采样的图像目录"

        # 下采样图像
        downscale_images(source_images)

        progress(1.0, desc="下采样完成！")
        update_progress("下采样", 100, 100, "多尺度下采样完成（2x、4x）")
        
        processing_active = False
        return f"✅ 下采样完成！生成了2x、4x尺度的图像"
        
    except Exception as e:
        processing_active = False
        return f"❌ 下采样失败：{str(e)}"

def detect_camera_model(input_folder: str) -> str:
    """检测使用的相机模型"""
    # 简化版本：通过检查sparse/0目录来判断
    # 在实际使用中，可以根据需要扩展这个函数
    try:
        output_path = Path(input_folder)
        sparse_0_path = output_path / "sparse" / "0" / "cameras.bin"
        if sparse_0_path.exists():
            # 这里可以进一步解析cameras.bin文件来确定相机模型
            # 现在暂时返回默认值
            pass
    except Exception:
        pass
    
    return "OPENCV"  # 默认返回OPENCV，用户可以在界面中手动选择

def step_estimate_depth(input_folder: str, output_folder: str, progress=gr.Progress()):
    """步骤5：估计数据集深度"""
    global processing_active
    
    error_msg = check_step_prerequisites("estimate_depth", input_folder)
    if error_msg:
        return error_msg
    
    processing_active = True
    
    try:
        # 检测相机模型
        camera_model = detect_camera_model(input_folder)
        
        # 智能选择图像目录进行深度估计
        output_path = Path(output_folder)
        
        def select_best_image_dir():
            """选择最适合的图像目录进行深度估计"""
            candidates = [
                (output_path / "undistorted" / "images_2", str(output_path / "undistorted"), "images_2"),
                (output_path / "undistorted" / "images_4", str(output_path / "undistorted"), "images_4"), 
                (output_path / "images_2", str(output_path), "images_2"),  # 当前目录下采样
                (output_path / "images_4", str(output_path), "images_4"),  # 当前目录下采样
                (output_path / "undistorted" / "images", str(output_path / "undistorted"), "images"),
                (output_path / "images", str(output_path), "images")
            ]
            
            for path, dataset_path, image_dir in candidates:
                if path.exists():
                    image_files = list(path.glob('*.png')) + list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
                    if image_files:
                        return dataset_path, image_dir, len(image_files)
            
            return None, None, 0
        
        dataset_path, image_dir, image_count = select_best_image_dir()
        
        if not dataset_path:
            processing_active = False
            return "❌ 深度估计失败: 找不到可用于深度估计的图像文件"
        
        if camera_model == "SPHERE":
            # Sphere模式使用PanDA
            progress(0.1, desc=f"Sphere模式 - 使用图像目录: {image_dir} ({image_count}张图像)")
            update_progress("深度估计", 10, 100, f"Sphere模式 - 使用图像目录: {image_dir}")
            
            progress(0.3, desc="正在使用PanDA估计深度...")
            update_progress("深度估计", 30, 100, "运行PanDA模型")
            
            result = estimate_depth_panda(str(dataset_path), str(image_dir))
        else:
            # 普通模式使用Depth Anything V2
            progress(0.1, desc=f"使用图像目录: {image_dir} ({image_count}张图像)")
            update_progress("深度估计", 10, 100, f"使用图像目录: {image_dir}")
            
            progress(0.3, desc="正在估计深度...")
            update_progress("深度估计", 30, 100, "运行Depth Anything V2模型")
            
            result = estimate_depth(str(dataset_path), str(image_dir))
        
        progress(1.0, desc="深度估计完成！")
        update_progress("深度估计", 100, 100, "深度估计完成")
        
        processing_active = False
        return f"✅ {result}"
        
    except Exception as e:
        processing_active = False
        update_progress("深度估计", 0, 100, f"深度估计失败: {str(e)}")
        return f"❌ 深度估计失败: {str(e)}"


def step_generate_mask(input_folder: str, output_folder: str, min_depth: float = 0.95, max_depth: float = 1.0, progress=gr.Progress()):
    """生成深度mask步骤"""
    try:
        global processing_active
        if not processing_active:
            return "处理已被中止"
        
        output_path = Path(output_folder)
        
        # 查找深度图目录
        depth_dir = output_path / "estimated_depths"
        if not depth_dir.exists():
            return "❌ Mask生成失败: 未找到深度估计结果目录"
        
        update_progress("生成Mask", 10, 100, "查找图像目录")
        
        # 智能选择图像目录进行mask生成
        def select_best_image_dir():
            """选择最适合的图像目录进行mask生成"""
            candidates = [
                (output_path / "undistorted" / "images", "undistorted/images"),
                (output_path / "images", "images"),
            ]
            
            for path, desc in candidates:
                if path.exists():
                    image_files = list(path.glob('*.png')) + list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
                    if image_files:
                        return path, desc, len(image_files)
            
            return None, None, 0
        
        image_dir, image_desc, image_count = select_best_image_dir()
        
        if not image_dir:
            return "❌ Mask生成失败: 找不到可用于生成mask的图像文件"
        
        update_progress("生成Mask", 30, 100, f"使用图像目录: {image_desc} ({image_count}张图像)")
        
        # 设置mask输出目录
        mask_dir = output_path / "masks"
        
        # 检查是否存在下采样图像
        has_downsampled = False
        scales = []
        for scale in [2, 4]:
            if (output_path / f"images_{scale}").exists() or (output_path / "undistorted" / f"images_{scale}").exists():
                has_downsampled = True
                scales.append(scale)
        
        progress(0.5, desc="生成深度mask...")
        update_progress("生成Mask", 50, 100, f"深度范围: [{min_depth:.2f}, {max_depth:.2f}], 反转mask")
        
        if has_downsampled and scales:
            # 生成原分辨率和下采样mask
            generate_masks_with_downsampling(
                base_image_dir=image_dir,
                depth_dir=depth_dir,
                output_base_dir=mask_dir,
                scales=scales,
                min_depth=min_depth,
                max_depth=max_depth,
                mask_suffix="_mask",
                invert_mask=True  # 反转mask，近景为0
            )
            update_progress("生成Mask", 90, 100, f"已生成原分辨率mask和{len(scales)}个下采样mask")
        else:
            # 只生成原分辨率mask
            process_image_directory(
                image_dir=image_dir,
                depth_dir=depth_dir,
                output_dir=mask_dir,
                min_depth=min_depth,
                max_depth=max_depth,
                mask_suffix="_mask",
                invert_mask=True  # 反转mask，近景为0
            )
            update_progress("生成Mask", 90, 100, "已生成原分辨率mask")
        
        progress(1.0, desc="Mask生成完成！")
        update_progress("生成Mask", 100, 100, "Mask生成完成")
        
        mask_count = len(list(mask_dir.glob("*_mask.png"))) if mask_dir.exists() else 0
        result_msg = f"Mask生成完成，共生成{mask_count}个mask文件，结果保存在: {mask_dir}"
        
        if has_downsampled and scales:
            result_msg += f"\n同时生成了{len(scales)}个下采样倍数的mask"
        
        return result_msg
        
    except Exception as e:
        update_progress("生成Mask", 0, 100, f"Mask生成失败: {str(e)}")
        return f"❌ Mask生成失败: {str(e)}"


def step_downscale_masks(input_folder: str, output_folder: str, progress=gr.Progress()):
    """步骤6：下采样masks"""
    global processing_active
    
    error_msg = check_step_prerequisites("downscale_masks", input_folder)
    if error_msg:
        return error_msg
    
    processing_active = True
    
    try:
        output_path = Path(output_folder)
        undistorted_masks = output_path / "undistorted/masks"
        original_masks = output_path / "masks"
        
        progress(0.1, desc="正在下采样masks...")
        update_progress("下采样masks", 10, 100, "开始下采样masks")
        
        # 选择下采样源目录
        if undistorted_masks.exists():
            source_masks = undistorted_masks
            update_progress("下采样masks", 20, 100, "从去畸变masks进行下采样")
        elif original_masks.exists():
            source_masks = original_masks
            update_progress("下采样masks", 20, 100, "从原始masks进行下采样")
        else:
            processing_active = False
            return "错误：找不到可下采样的masks目录"
        
        # 下采样masks
        downscale_masks_dir(source_masks)
        
        progress(1.0, desc="下采样masks完成！")
        update_progress("下采样masks", 100, 100, "masks下采样完成")
        
        processing_active = False
        return f"✅ 下采样masks完成！生成了2x、4x尺度的masks"
        
    except Exception as e:
        processing_active = False
        return f"❌ 下采样masks失败：{str(e)}"


def process_selected_steps(input_folder, total_frames, matching_method: Literal["vocab_tree", "exhaustive", "sequential"], refine_intrinsics, camera_model,
                          enable_extract, enable_colmap, enable_undistort, enable_downscale, enable_estimate_depth, enable_generate_mask, 
                          min_depth, max_depth, enable_downscale_masks, progress=gr.Progress()):
    """根据勾选状态处理选定的步骤"""
    global processing_active
    
    if not input_folder:
        return "错误：请选择输入文件夹"
    
    # 输出文件夹设置为输入文件夹
    output_folder = input_folder
    
    # 检查至少选择了一个步骤
    selected_steps = []
    if enable_extract:
        selected_steps.append("extract_frames")
    if enable_colmap:
        selected_steps.append("colmap")
    if enable_undistort:
        selected_steps.append("undistort")
    if enable_downscale:
        selected_steps.append("downscale")
    if enable_estimate_depth:
        selected_steps.append("estimate_depth")
    if enable_generate_mask:
        selected_steps.append("generate_mask")
    if enable_downscale_masks:
        selected_steps.append("downscale_masks")
    
    if not selected_steps:
        return "错误：请至少选择一个处理步骤"
    
    processing_active = True
    results = []
    
    try:
        output_path = Path(output_folder)
        total_steps = len(selected_steps)
        
        # 步骤1：提取视频帧
        if enable_extract:
            current_step = selected_steps.index("extract_frames") + 1
            progress(current_step / total_steps * 0.25, desc=f"步骤 {current_step}/{total_steps}: 正在提取视频帧...")
            
            error_msg = check_step_prerequisites("extract_frames", input_folder)
            if error_msg:
                processing_active = False
                return error_msg
            
            update_progress("提取视频帧", int(current_step / total_steps * 25), 100, "开始提取视频帧")
            
            def progress_callback(message):
                update_progress("提取视频帧", int(current_step / total_steps * 25), 100, message)
            
            images_path = output_path / "images"
            extracted_count = extract_frames_with_progress(input_folder, images_path, total_frames, progress_callback)
            results.append(f"✅ 视频帧提取完成！共提取了 {extracted_count} 帧")
        
        # 步骤2：相机标定
        if enable_colmap:
            current_step = selected_steps.index("colmap") + 1
            progress(current_step / total_steps * 0.5, desc=f"步骤 {current_step}/{total_steps}: 正在进行相机标定...")
            
            error_msg = check_step_prerequisites("colmap", input_folder)
            if error_msg:
                processing_active = False
                return error_msg
            
            update_progress("相机标定", int(current_step / total_steps * 50), 100, f"开始COLMAP相机标定 (相机模型: {camera_model})")
            
            images_path = output_path / "images"
            run_colmap_with_camera_model(
                image_dir=images_path,
                colmap_dir=output_path,
                matching_method=matching_method,
                refine_intrinsics=refine_intrinsics,
                camera_model=camera_model,
            )
            results.append(f"✅ 相机标定完成！使用相机模型：{camera_model}")
        
        # 步骤3：去畸变
        if enable_undistort:
            current_step = selected_steps.index("undistort") + 1
            progress(current_step / total_steps * 0.75, desc=f"步骤 {current_step}/{total_steps}: 正在去畸变...")
            
            if camera_model == "SPHERE":
                # Sphere模式下不进行去畸变，直接创建软链接
                update_progress("去畸变", int(current_step / total_steps * 75), 100, "Sphere模式：跳过去畸变，创建目录结构")
                
                images_path = output_path / "images"
                undistorted_path = output_path / "undistorted"
                undistorted_images_path = undistorted_path / "images"
                
                # 创建undistorted目录结构
                undistorted_path.mkdir(exist_ok=True)
                if undistorted_images_path.exists():
                    import shutil
                    shutil.rmtree(undistorted_images_path)
                
                # 创建软链接或复制原始图像
                try:
                    undistorted_images_path.symlink_to(images_path.resolve())
                except OSError:
                    # 如果软链接失败，复制文件
                    import shutil
                    shutil.copytree(images_path, undistorted_images_path)
                
                # 复制sparse目录到undistorted
                undistorted_sparse_path = undistorted_path / "sparse"
                if undistorted_sparse_path.exists():
                    import shutil
                    shutil.rmtree(undistorted_sparse_path)
                import shutil
                shutil.copytree(output_path / "sparse", undistorted_sparse_path)
                
                results.append("✅ Sphere模式：跳过去畸变，目录结构已创建！")
            else:
                # 普通模式进行去畸变
                error_msg = check_step_prerequisites("undistort", input_folder)
                if error_msg:
                    processing_active = False
                    return error_msg
                
                update_progress("去畸变", int(current_step / total_steps * 75), 100, "开始图像去畸变")
                
                run_undistort(output_path)
                results.append("✅ 去畸变完成！")
        
        # 步骤4：下采样
        if enable_downscale:
            current_step = selected_steps.index("downscale") + 1
            progress_ratio = current_step / total_steps
            progress(progress_ratio * 0.9, desc=f"步骤 {current_step}/{total_steps}: 正在下采样...")
            
            error_msg = check_step_prerequisites("downscale", input_folder)
            if error_msg:
                processing_active = False
                return error_msg
            
            update_progress("下采样", int(progress_ratio * 90), 100, "开始多尺度下采样")
            
            # 确定下采样的源目录和输出位置
            undistorted_images = output_path / "undistorted/images"
            original_images = output_path / "images"
            
            if undistorted_images.exists():
                # 有去畸变目录，在undistorted下进行下采样
                source_images = undistorted_images
                update_progress("下采样", int(progress_ratio * 90), 100, "从去畸变图像进行下采样")
                downscale_images(source_images)
            elif original_images.exists():
                # 没有去畸变目录，直接在当前目录进行下采样
                source_images = original_images
                update_progress("下采样", int(progress_ratio * 90), 100, "从原始图像进行下采样（保存到当前目录）")
                downscale_images(source_images)
            else:
                processing_active = False
                return "错误：找不到可下采样的图像目录"
            results.append("✅ 下采样完成！（2x、4x）")
        
        # 步骤5：深度估计
        if enable_estimate_depth:
            current_step = selected_steps.index("estimate_depth") + 1
            progress_ratio = current_step / total_steps
            progress(progress_ratio * 0.9, desc=f"步骤 {current_step}/{total_steps}: 正在估计深度...")
            
            error_msg = check_step_prerequisites("estimate_depth", input_folder)
            if error_msg:
                processing_active = False
                return error_msg
            
            # 智能选择图像目录进行深度估计 (优先使用原分辨率)
            def select_best_image_dir():
                """选择最适合的图像目录进行深度估计，优先使用原分辨率"""
                candidates = [
                    (output_path / "undistorted" / "images", str(output_path / "undistorted"), "images"),
                    (output_path / "images", str(output_path), "images"),
                    (output_path / "undistorted" / "images_2", str(output_path / "undistorted"), "images_2"),
                    (output_path / "images_2", str(output_path), "images_2"),  # 当前目录下采样
                    (output_path / "undistorted" / "images_4", str(output_path / "undistorted"), "images_4"), 
                    (output_path / "images_4", str(output_path), "images_4"),  # 当前目录下采样
                ]
                
                for path, dataset_path, image_dir in candidates:
                    if path.exists():
                        image_files = list(path.glob('*.png')) + list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
                        if image_files:
                            return dataset_path, image_dir, len(image_files)
                
                return None, None, 0
            
            dataset_path, image_dir, image_count = select_best_image_dir()
            
            if not dataset_path:
                processing_active = False
                return "错误：找不到可用于深度估计的图像文件"
            
            if camera_model == "SPHERE":
                # Sphere模式使用PanDA进行深度估计
                update_progress("深度估计", int(progress_ratio * 90), 100, f"Sphere模式：PanDA深度估计 - {image_dir} ({image_count}张图像)")
                result = estimate_depth_panda(str(dataset_path), str(image_dir))
                results.append(f"✅ {result}")
            else:
                # 普通模式使用Depth Anything V2
                update_progress("深度估计", int(progress_ratio * 90), 100, f"深度估计 - {image_dir} ({image_count}张图像)")
                result = estimate_depth(str(dataset_path), str(image_dir))
                results.append(f"✅ {result}")
        
        # 步骤6：生成深度mask
        if enable_generate_mask:
            current_step = selected_steps.index("generate_mask") + 1
            progress_ratio = current_step / total_steps
            progress(progress_ratio * 0.9, desc=f"步骤 {current_step}/{total_steps}: 正在生成深度mask...")
            
            error_msg = check_step_prerequisites("generate_mask", input_folder)
            if error_msg:
                processing_active = False
                return error_msg
            
            update_progress("生成Mask", int(progress_ratio * 90), 100, f"生成深度mask - 深度范围: [{min_depth:.2f}, {max_depth:.2f}]")
            result = step_generate_mask(input_folder, output_folder, min_depth, max_depth, progress)
            if result.startswith("❌"):
                processing_active = False
                return result
            results.append(f"✅ {result}")
        
        # 步骤7：下采样masks
        if enable_downscale_masks:
            current_step = selected_steps.index("downscale_masks") + 1
            progress_ratio = current_step / total_steps
            progress(progress_ratio * 0.9, desc=f"步骤 {current_step}/{total_steps}: 正在下采样masks...")
            
            error_msg = check_step_prerequisites("downscale_masks", input_folder)
            if error_msg:
                processing_active = False
                return error_msg
            
            update_progress("下采样masks", int(progress_ratio * 90), 100, "开始下采样masks")
            
            # 确定下采样的源目录
            undistorted_masks = output_path / "undistorted/masks"
            original_masks = output_path / "masks"
            
            if undistorted_masks.exists():
                source_masks = undistorted_masks
                update_progress("下采样masks", int(progress_ratio * 95), 100, "从去畸变masks进行下采样")
                downscale_masks_dir(source_masks)
            elif original_masks.exists():
                source_masks = original_masks
                update_progress("下采样masks", int(progress_ratio * 95), 100, "从原始masks进行下采样")
                downscale_masks_dir(source_masks)
            else:
                processing_active = False
                return "错误：找不到可下采样的masks目录"
            
            results.append("✅ 下采样masks完成！（2x、4x）")
        
        # 所有步骤完成
        progress(1.0, desc="所有步骤完成！")
        update_progress("完成", 100, 100, "所有选定的步骤已完成")
        
        processing_active = False
        return f"🎉 选定步骤处理完成！\n\n" + "\n".join(results) + f"\n\n输出文件保存在：{output_folder}"
        
    except Exception as e:
        processing_active = False
        return f"❌ 处理过程中出现错误：{str(e)}"

def get_progress_info():
    """获取当前进度信息"""
    if processing_active:
        return f"当前步骤：{progress_info['step']} - {progress_info['message']}"
    return "等待处理..."

def validate_colmap():
    """验证COLMAP是否可用"""
    try:
        # 直接检查colmap命令是否可用
        result = subprocess.run(["colmap", "-h"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 尝试获取版本信息
            for line in result.stdout.split("\n"):
                if line.startswith("COLMAP"):
                    version = line.split(" ")[1]
                    return f"✅ COLMAP 版本 {version} 可用"
            return "✅ COLMAP 可用"
        else:
            return "❌ COLMAP 未找到，请确保已安装并添加到PATH"
    except Exception as e:
        return f"❌ COLMAP 未找到，请确保已安装并添加到PATH (错误: {str(e)})"

def validate_ffmpeg():
    """验证FFmpeg是否可用"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            return "✅ FFmpeg 可用"
        else:
            return "❌ FFmpeg 未找到"
    except:
        return "❌ FFmpeg 未找到，请确保已安装并添加到PATH"

def browse_folder(current_dir=""):
    """使用tkinter调用系统文件浏览器"""
    try:
        # 创建隐藏的tkinter窗口
        root = tk.Tk()
        root.withdraw()
        
        # 确定初始目录
        if current_dir and os.path.exists(current_dir):
            initial_dir = current_dir
        elif os.path.exists("/mnt"):
            initial_dir = "/mnt"
        else:
            initial_dir = os.path.expanduser("~")
        
        # 打开文件夹选择对话框
        folder_path = filedialog.askdirectory(
            title="选择文件夹",
            initialdir=initial_dir
        )
        
        # 关闭tkinter窗口
        root.destroy()
        
        return folder_path if folder_path else ""
    
    except Exception as e:
        print(f"文件浏览器错误: {e}")
        return ""

def create_interface():
    """创建gradio界面"""
    with gr.Blocks(title="视频帧处理工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎥 视频帧处理工具")
        gr.Markdown("这个工具可以从视频中提取帧，进行相机标定，去畸变和下采样处理。所有处理结果将保存在视频文件夹中。")
        
        with gr.Tab("处理设置"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_folder = gr.Textbox(
                            label="输入视频文件夹",
                            placeholder="选择包含视频文件的文件夹路径",
                            interactive=True,
                            scale=4
                        )
                        input_browse_btn = gr.Button("🗂️ 浏览", scale=1)
                    

                    
                    total_frames = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        value=100,
                        step=10,
                        label="总帧数",
                        info="从视频中提取的总帧数"
                    )
                    
                    matching_method = gr.Dropdown(
                        choices=["vocab_tree", "exhaustive", "sequential"],
                        value="vocab_tree",
                        label="匹配方法",
                        info="COLMAP特征匹配方法"
                    )
                    
                    camera_model = gr.Dropdown(
                        choices=["PINHOLE", "OPENCV", "SPHERE"],
                        value="OPENCV",
                        label="相机模型",
                        info="COLMAP相机模型选择"
                    )
                    
                    refine_intrinsics = gr.Checkbox(
                        value=True,
                        label="细化内参",
                        info="是否在bundle adjustment中细化相机内参"
                    )
                
                with gr.Column():
                    # 系统检查
                    gr.Markdown("### 系统检查")
                    colmap_status = gr.Textbox(
                        value=validate_colmap(),
                        label="COLMAP 状态",
                        interactive=False
                    )
                    
                    ffmpeg_status = gr.Textbox(
                        value=validate_ffmpeg(),
                        label="FFmpeg 状态",
                        interactive=False
                    )
                    
                    # 刷新按钮
                    refresh_btn = gr.Button("刷新状态", variant="secondary")
                    
                    def refresh_status():
                        return validate_colmap(), validate_ffmpeg()
                    
                    refresh_btn.click(
                        refresh_status,
                        outputs=[colmap_status, ffmpeg_status]
                    )
            
            # 绑定浏览按钮
            # 添加状态跟踪当前选择的目录
            current_input_dir = gr.State("/mnt" if os.path.exists("/mnt") else os.path.expanduser("~"))
            
            def browse_input(current_dir):
                path = browse_folder(current_dir)
                if path:
                    return path, path  # 返回新路径和更新的状态
                return gr.update(), current_dir  # 如果取消选择，保持原状态
            
            input_browse_btn.click(
                browse_input,
                inputs=[current_input_dir],
                outputs=[input_folder, current_input_dir]
            )
            
            # 当用户手动输入路径时，更新状态
            def update_input_dir_state(path):
                if path and os.path.exists(path):
                    return path
                return gr.update()
            
            input_folder.change(
                update_input_dir_state,
                inputs=[input_folder],
                outputs=[current_input_dir]
            )
            
            # 步骤选择
            gr.Markdown("### 📋 选择处理步骤")
            with gr.Row():
                with gr.Column():
                    enable_extract = gr.Checkbox(
                        value=True,
                        label="1️⃣ 提取视频帧",
                        info="从视频中提取帧图像"
                    )
                    
                    enable_colmap = gr.Checkbox(
                        value=True,
                        label="2️⃣ 相机标定",
                        info="使用COLMAP进行特征提取和匹配"
                    )
                
                with gr.Column():
                    enable_undistort = gr.Checkbox(
                        value=True,
                        label="3️⃣ 去畸变",
                        info="修正相机畸变（Sphere模式下不可用）"
                    )
                    
                    enable_downscale = gr.Checkbox(
                        value=True,
                        label="4️⃣ 下采样",
                        info="生成多个尺度的图像"
                    )
                    
                    enable_estimate_depth = gr.Checkbox(
                        value=False,
                        label="5️⃣ 深度估计",
                        info="使用Depth Anything V2估计深度（Sphere模式下使用PanDA）"
                    )
                    
                    enable_generate_mask = gr.Checkbox(
                        value=False,
                        label="6️⃣ 生成深度Mask",
                        info="基于深度图生成mask，用于过滤近景内容"
                    )
                
                with gr.Column():
                    enable_downscale_masks = gr.Checkbox(
                        value=False,
                        label="7️⃣ 下采样masks",
                        info="将生成的masks目录下采样到不同尺度"
                    )
            
            # 深度Mask参数设置
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 深度Mask参数")
                    min_depth = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        label="最小深度阈值",
                        info="近景深度范围的最小值 (inverse depth)"
                    )
                    max_depth = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.01,
                        label="最大深度阈值",
                        info="近景深度范围的最大值 (inverse depth)"
                    )
                    gr.Markdown("⚠️ **说明**: 深度范围内的区域将被mask为0（黑色），用于过滤拍摄者、三脚架等近景内容")
            
            # 快捷选择按钮
            with gr.Row():
                select_all_btn = gr.Button("✅ 全选", variant="secondary")
                select_none_btn = gr.Button("❌ 全不选", variant="secondary")
            
            def update_undistort_availability(camera_model_value):
                """根据相机模型更新去畸变选项的可用性"""
                if camera_model_value == "SPHERE":
                    # Sphere模式下禁用去畸变，并设置为False
                    return gr.update(value=False, interactive=False)
                else:
                    # 其他模式下启用去畸变
                    return gr.update(interactive=True)
            
            def select_all():
                # 注意：当用户点击全选时，去畸变的状态会根据相机模型自动调整
                return True, True, True, True, True, True, True
            
            def select_none():
                return False, False, False, False, False, False, False
            
            select_all_btn.click(
                select_all,
                outputs=[enable_extract, enable_colmap, enable_undistort, enable_downscale, enable_estimate_depth, enable_generate_mask, enable_downscale_masks]
            )
            
            select_none_btn.click(
                select_none,
                outputs=[enable_extract, enable_colmap, enable_undistort, enable_downscale, enable_estimate_depth, enable_generate_mask, enable_downscale_masks]
            )
            
            # 绑定相机模型变化事件
            camera_model.change(
                update_undistort_availability,
                inputs=[camera_model],
                outputs=[enable_undistort]
            )
            
            # 批量处理按钮
            gr.Markdown("### 🚀 批量处理")
            process_selected_btn = gr.Button("▶️ 处理选定步骤", variant="primary", size="lg")
            
            # 进度显示
            with gr.Row():
                progress_text = gr.Textbox(
                    label="处理进度",
                    value="等待处理...",
                    interactive=False
                )
            
            # 结果显示
            result_text = gr.Textbox(
                label="处理结果",
                interactive=False,
                lines=8
            )
            
            # 绑定批量处理函数
            process_selected_btn.click(
                process_selected_steps,
                inputs=[input_folder, total_frames, matching_method, refine_intrinsics, camera_model,
                       enable_extract, enable_colmap, enable_undistort, enable_downscale, enable_estimate_depth, enable_generate_mask,
                       min_depth, max_depth, enable_downscale_masks],
                outputs=[result_text]
            )
            
            # 定期更新进度
            def update_progress_display():
                return get_progress_info()
            
            demo.load(lambda: None, every=1).then(
                update_progress_display,
                outputs=[progress_text]
            )
        
        with gr.Tab("帮助信息"):
            gr.Markdown("""
            ## 使用说明
            
            ### 1. 准备工作
            - 确保系统中已安装 COLMAP 和 FFmpeg
            - 将视频文件放在一个文件夹中
            
            ### 2. 支持的视频格式
            - MP4 (.mp4)
            - AVI (.avi)
            - MOV (.mov)
            - MKV (.mkv)
            
            ### 3. 处理模式
            
            #### 📋 **步骤选择模式**
            - 勾选需要执行的步骤
            - 点击"处理选定步骤"批量执行
            - 支持任意步骤组合
            
            #### ⚡ **快捷选择**
            - **全选**：选择所有步骤
            - **全不选**：取消所有选择
            
            ### 4. 处理步骤
            - **1️⃣ 提取视频帧**：从视频中提取帧图像
            - **2️⃣ 相机标定**：使用COLMAP进行特征提取和匹配
            - **3️⃣ 去畸变**：修正相机畸变
            - **4️⃣ 下采样**：生成多个尺度的图像
            - **5️⃣ 深度估计**：使用深度估计模型生成深度图（默认使用原分辨率）
            - **6️⃣ 生成深度Mask**：基于深度图生成mask，用于过滤近景内容
            - **7️⃣ 下采样masks**：将生成的masks目录下采样到不同尺度
            
            ### 5. 参数说明
            - **总帧数**：从所有视频中提取的总帧数，会根据视频时长自动分配
            - **匹配方法**：
              - `vocab_tree`：使用词汇树匹配（推荐，速度快）
              - `exhaustive`：穷举匹配（精度高，速度慢）
              - `sequential`：顺序匹配（适合序列图像）
            - **相机模型**：
              - `OPENCV_FISHEYE`：鱼眼相机模型（推荐）
              - `OPENCV`：OpenCV标准相机模型
              - `PINHOLE`：针孔相机模型
              - `RADIAL`：径向畸变模型
              - `SIMPLE_PINHOLE`：简单针孔模型
              - `SIMPLE_RADIAL`：简单径向畸变模型
              - `FULL_OPENCV`：完整OpenCV模型
            - **细化内参**：是否在束集调整中优化相机内参
            - **深度范围**：用于生成mask的深度阈值
              - `最小深度阈值`：近景区域的最小深度值（inverse depth）
              - `最大深度阈值`：近景区域的最大深度值（inverse depth）
              - 范围内的区域将被mask为0，用于过滤拍摄者、三脚架等近景内容
            
            ### 6. 文件浏览器使用
            - 点击"🗂️ 浏览"按钮打开系统文件浏览器
            - 默认从 `/mnt` 目录开始浏览（WSL环境推荐）
            - 文件浏览器会记住上次选择的目录，再次点击时从该目录开始
            - 手动输入路径时，浏览器状态会自动更新
            - 在弹出的对话框中选择目标文件夹
            - 点击"选择文件夹"确认选择
            - 支持WSL环境下的图形界面
            
            ### 7. 输出结果
            处理完成后，视频文件夹将包含：
            - `images/`：提取的原始帧
            - `sparse/`：COLMAP稀疏重建结果
            - `undistorted/`：去畸变后的图像和相机参数
            - `images_2/`, `images_4/`：不同尺度的下采样图像
            - `estimated_depths/`：深度估计结果（.npy格式）
            - `masks/`, `masks_2/`, `masks_4/`：深度mask（原分辨率和下采样）
            
            ### 8. 使用技巧
            - 可以根据需要选择部分步骤执行
            - 步骤之间有依赖关系，会自动检查前置条件
            - 如果某步骤失败，可以重新选择并执行
            - 批量处理支持进度显示和详细的结果报告
            - 提取视频帧时会显示详细的帧级别进度
            - 根据视频设备类型选择合适的相机模型
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 