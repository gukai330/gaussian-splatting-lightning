#!/usr/bin/env python3
"""
深度图mask生成脚本
用于载入depthmap，选择depth范围，生成binary mask
主要用于去除360图像中的近景内容
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from tqdm import tqdm
import cv2


def load_depth_map(depth_path: str) -> np.ndarray:
    """
    载入npy格式的depth map
    
    Args:
        depth_path: depth map文件路径
        
    Returns:
        深度图数组 (0-1的inverse depth)
    """
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"深度图文件不存在: {depth_path}")
    
    depth_map = np.load(depth_path)
    
    # 确保深度图在0-1范围内
    if depth_map.max() > 1.0 or depth_map.min() < 0.0:
        print(f"警告: 深度图不在0-1范围内，当前范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        depth_map = np.clip(depth_map, 0.0, 1.0)
    
    return depth_map


def resize_depth_to_image(depth_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    将深度图resize到目标图像尺寸
    
    Args:
        depth_map: 原始深度图
        target_size: 目标尺寸 (width, height)
        
    Returns:
        调整后的深度图
    """
    resized_depth = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_depth


def generate_depth_mask(depth_map: np.ndarray, 
                       min_depth: float = 0.0, 
                       max_depth: float = 1.0,
                       invert_mask: bool = False) -> np.ndarray:
    """
    根据深度范围生成binary mask
    
    Args:
        depth_map: 深度图 (inverse depth, 0-1范围)
        min_depth: 最小深度阈值
        max_depth: 最大深度阈值
        invert_mask: 是否反转mask (默认False，保留指定范围内的区域)
        
    Returns:
        Binary mask (0或255)
    """
    # 创建mask: 在指定深度范围内的像素为True
    mask = (depth_map >= min_depth) & (depth_map <= max_depth)
    
    if invert_mask:
        mask = ~mask
    
    # 转换为0-255的uint8格式
    binary_mask = mask.astype(np.uint8) * 255
    
    return binary_mask


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    获取图像尺寸
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像尺寸 (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def process_image_directory(image_dir: Path, 
                          depth_dir: Path,
                          output_dir: Path,
                          min_depth: float = 0.0,
                          max_depth: float = 0.3,
                          mask_suffix: str = "_mask",
                          invert_mask: bool = False) -> None:
    """
    处理整个图像目录，为每张图像生成对应的mask
    
    Args:
        image_dir: 图像目录路径
        depth_dir: 深度图目录路径
        output_dir: mask输出目录路径
        min_depth: 最小深度阈值
        max_depth: 最大深度阈值
        mask_suffix: mask文件名后缀
        invert_mask: 是否反转mask
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    
    for image_file in tqdm(image_files, desc="处理图像"):
        try:
            # 构建对应的深度图文件路径 (图像文件名 + .npy)
            depth_file = depth_dir / f"{image_file.name}.npy"
            
            if not depth_file.exists():
                print(f"警告: 未找到对应的深度图文件 {depth_file}")
                continue
            
            # 载入深度图
            depth_map = load_depth_map(str(depth_file))
            
            # 获取图像尺寸
            image_size = get_image_size(str(image_file))
            
            # 调整深度图尺寸到图像尺寸
            depth_resized = resize_depth_to_image(depth_map, image_size)
            
            # 生成mask
            mask = generate_depth_mask(depth_resized, min_depth, max_depth, invert_mask)
            
            # 保存mask
            mask_filename = f"{image_file.stem}{mask_suffix}.png"
            mask_path = output_dir / mask_filename
            
            cv2.imwrite(str(mask_path), mask)
            
        except Exception as e:
            print(f"处理文件 {image_file} 时出错: {e}")


def generate_masks_with_downsampling(base_image_dir: Path,
                                   depth_dir: Path,
                                   output_base_dir: Path,
                                   scales: List[int] = [2, 4],
                                   min_depth: float = 0.0,
                                   max_depth: float = 0.3,
                                   mask_suffix: str = "_mask",
                                   invert_mask: bool = False) -> None:
    """
    为原始图像和下采样图像生成mask
    确保所有mask都基于原图像分辨率生成，然后下采样
    
    Args:
        base_image_dir: 基础图像目录路径
        depth_dir: 深度图目录路径  
        output_base_dir: mask输出基础目录路径
        scales: 下采样倍数列表
        min_depth: 最小深度阈值
        max_depth: 最大深度阈值
        mask_suffix: mask文件名后缀
        invert_mask: 是否反转mask
    """
    
    # 首先处理原始图像生成原分辨率mask
    print("处理原始尺寸图像...")
    process_image_directory(
        image_dir=base_image_dir,
        depth_dir=depth_dir,
        output_dir=output_base_dir,
        min_depth=min_depth,
        max_depth=max_depth,
        mask_suffix=mask_suffix,
        invert_mask=invert_mask
    )
    
    # 为下采样图像生成对应分辨率的mask
    for scale in scales:
        downscale_image_dir = base_image_dir.parent / f"images_{scale}"
        downscale_output_dir = output_base_dir.parent / f"masks_{scale}"
        
        if downscale_image_dir.exists():
            print(f"为 {scale}x 下采样图像生成对应mask...")
            generate_downsampled_masks(
                original_mask_dir=output_base_dir,
                downscale_image_dir=downscale_image_dir,
                downscale_output_dir=downscale_output_dir,
                scale=scale,
                mask_suffix=mask_suffix
            )
        else:
            print(f"下采样图像目录不存在: {downscale_image_dir}")


def generate_downsampled_masks(original_mask_dir: Path,
                             downscale_image_dir: Path,
                             downscale_output_dir: Path,
                             scale: int,
                             mask_suffix: str = "_mask") -> None:
    """
    基于原分辨率mask生成下采样mask
    
    Args:
        original_mask_dir: 原分辨率mask目录
        downscale_image_dir: 下采样图像目录
        downscale_output_dir: 下采样mask输出目录
        scale: 下采样倍数
        mask_suffix: mask文件后缀
    """
    downscale_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有下采样图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    downscale_image_files = [f for f in downscale_image_dir.iterdir() 
                           if f.suffix.lower() in image_extensions]
    
    if not downscale_image_files:
        print(f"在目录 {downscale_image_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(downscale_image_files)} 张下采样图像，生成对应mask...")
    
    for image_file in tqdm(downscale_image_files, desc=f"生成{scale}x下采样mask"):
        try:
            # 查找对应的原分辨率mask
            original_mask_path = original_mask_dir / f"{image_file.stem}{mask_suffix}.png"
            
            if not original_mask_path.exists():
                print(f"警告: 未找到对应的原分辨率mask文件 {original_mask_path}")
                continue
            
            # 载入原分辨率mask
            original_mask = cv2.imread(str(original_mask_path), cv2.IMREAD_GRAYSCALE)
            if original_mask is None:
                print(f"警告: 无法读取mask文件 {original_mask_path}")
                continue
            
            # 获取下采样图像的尺寸
            downscale_image_size = get_image_size(str(image_file))
            
            # 将mask下采样到对应尺寸
            downsampled_mask = cv2.resize(original_mask, downscale_image_size, interpolation=cv2.INTER_NEAREST)
            
            # 保存下采样mask
            downscale_mask_path = downscale_output_dir / f"{image_file.stem}{mask_suffix}.png"
            cv2.imwrite(str(downscale_mask_path), downsampled_mask)
            
        except Exception as e:
            print(f"处理文件 {image_file} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="从深度图生成binary mask，用于去除360图像中的近景内容"
    )
    
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="图像目录路径"
    )
    
    parser.add_argument(
        "--depth_dir", 
        type=str, 
        required=True,
        help="深度图目录路径 (包含.npy文件)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="mask输出目录路径"
    )
    
    parser.add_argument(
        "--min_depth", 
        type=float, 
        default=0.0,
        help="最小深度阈值 (inverse depth, 默认: 0.0)"
    )
    
    parser.add_argument(
        "--max_depth", 
        type=float, 
        default=0.3,
        help="最大深度阈值 (inverse depth, 默认: 0.3，用于过滤近景)"
    )
    
    parser.add_argument(
        "--mask_suffix", 
        type=str, 
        default="_mask",
        help="mask文件名后缀 (默认: _mask)"
    )
    
    parser.add_argument(
        "--invert_mask", 
        action="store_true",
        help="反转mask (保留指定范围外的区域)"
    )
    
    parser.add_argument(
        "--enable_downsampling", 
        action="store_true",
        help="是否为下采样图像也生成mask"
    )
    
    parser.add_argument(
        "--scales", 
        type=int, 
        nargs="+", 
        default=[2, 4],
        help="下采样倍数列表 (默认: 2 4)"
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    image_dir = Path(args.image_dir)
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)
    
    # 验证输入目录
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    
    if not depth_dir.exists():
        raise FileNotFoundError(f"深度图目录不存在: {depth_dir}")
    
    print(f"图像目录: {image_dir}")
    print(f"深度图目录: {depth_dir}")
    print(f"输出目录: {output_dir}")
    print(f"深度范围: [{args.min_depth}, {args.max_depth}]")
    print(f"反转mask: {args.invert_mask}")
    print(f"下采样处理: {args.enable_downsampling}")
    if args.enable_downsampling:
        print(f"下采样倍数: {args.scales}")
    
    # 执行处理
    if args.enable_downsampling:
        generate_masks_with_downsampling(
            base_image_dir=image_dir,
            depth_dir=depth_dir,
            output_base_dir=output_dir,
            scales=args.scales,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            mask_suffix=args.mask_suffix,
            invert_mask=args.invert_mask
        )
    else:
        process_image_directory(
            image_dir=image_dir,
            depth_dir=depth_dir,
            output_dir=output_dir,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            mask_suffix=args.mask_suffix,
            invert_mask=args.invert_mask
        )
    
    print("处理完成！")


if __name__ == "__main__":
    main() 