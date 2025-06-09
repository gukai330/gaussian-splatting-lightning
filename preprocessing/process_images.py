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


def run_command(cmd: str, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
       
        print(f"[bold red]Error running command: {cmd}")
        print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


def extract_frames_ffmpeg(input_folder, output_folder, total_frames=10):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    durations = []
    for file in video_files:
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
    alloc = {f: max(1, int(total_frames * (d / total_duration))) for f, d in durations}
    for file, dur in tqdm(durations, desc="Extracting frames"):
        if dur == 0: continue
        video_path = os.path.join(input_folder, file)
        prefix = os.path.join(output_folder, Path(file).stem)
        frames = alloc[file]
        timestamps = np.linspace(0, dur, frames)
        for i, t in enumerate(timestamps):
            outname = f"{prefix}_frame{i}.png"
            cmd = f"ffmpeg -ss {t:.2f} -i \"{video_path}\" -frames:v 1 -pix_fmt rgb24 -y \"{outname}\""
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_colmap_version(colmap_cmd: str = "colmap", default_version: str = "3.8") -> Version:
    output = run_command(f"{colmap_cmd} -h", verbose=False)
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            return Version(line.split(" ")[1])
    return Version(default_version)


def get_vocab_tree() -> Path:
    path = Path(os.path.expanduser("~/.cache/nerfstudio/vocab_tree.fbow"))
    if not path.exists():
        url = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, stream=True)
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f:
            for chunk in tqdm(r.iter_content(1024), total=total // 1024 + 1, desc="Downloading vocab_tree"):
                f.write(chunk)
    return path


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "sequential",
    refine_intrinsics: bool = True,
    colmap_cmd: str = "colmap",
):
    db = colmap_dir / "database.db"
    db.unlink(missing_ok=True)
    version = get_colmap_version(colmap_cmd)

    # Feature Extraction
    run_command(f"{colmap_cmd} feature_extractor "
                f"--database_path {db} "
                f"--image_path {image_dir} "
                f"--ImageReader.single_camera 1 "
                f"--SiftExtraction.use_gpu 1 "
                f"--ImageReader.camera_model OPENCV_FISHEYE", verbose=True)

    # Feature Matching
    match_cmd = f"{colmap_cmd} {matching_method}_matcher --database_path {db} --SiftMatching.use_gpu 1"
    if matching_method == "vocab_tree":
        vocab = get_vocab_tree()
        match_cmd += f" --VocabTreeMatching.vocab_tree_path \"{vocab}\""
    run_command(match_cmd, True)

    # Mapping
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = (f"{colmap_cmd} mapper --database_path {db} "
                  f"--image_path {image_dir} "
                  f"--output_path {sparse_dir}")
    if version >= Version("3.7"):
        mapper_cmd += " --Mapper.ba_global_function_tolerance=1e-6"
    run_command(mapper_cmd, verbose=True)

    if refine_intrinsics:
        ba_cmd = f"{colmap_cmd} bundle_adjuster --input_path {sparse_dir}/0 --output_path {sparse_dir}/0 --BundleAdjustment.refine_principal_point 1"
        run_command(ba_cmd, verbose=True)


def run_undistort(output_dir: Path, colmap_cmd: str = "colmap"):
    """
    从多个 COLMAP 稀疏模型中自动选出图像最多的一个进行 undistort。
    """

    sparse_root = output_dir / "sparse"
    images = output_dir / "images"
    undistort_dir = output_dir / "undistorted"

    def count_registered_images(model_dir: Path):
        cmd = f"{colmap_cmd} model_analyzer --path \"{model_dir}\""
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return -1
        output = result.stdout.decode()
        for line in output.splitlines():
            if line.startswith("Registered images"):
                return int(line.split(":")[-1].strip())
        return -1

    # 搜索所有 sparse 子目录，找出图像最多的 model
    best_model_dir = None
    max_images = -1
    for sub in sparse_root.iterdir():
        if sub.is_dir():
            n = count_registered_images(sub)
            if n > max_images:
                max_images = n
                best_model_dir = sub

    if best_model_dir is None:
        raise RuntimeError("未找到有效的 COLMAP 稀疏模型目录！")

    print(f"[Info] 使用图像数最多的模型：{best_model_dir}（图像数：{max_images}）")

    # 运行 undistorter
    run_command(f"{colmap_cmd} image_undistorter "
                f"--image_path \"{images}\" "
                f"--input_path \"{best_model_dir}\" "
                f"--output_path \"{undistort_dir}\" "
                f"--output_type COLMAP "
                f"--max_image_size 2000")

def downscale_images(image_folder: Path, scales=[2, 4, 8, 16]):
    for s in scales:
        out = image_folder.parent / f"images_{s}"
        out.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(image_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(image_folder / f)
                img = img.resize((round(img.width / s), round(img.height / s)), Image.Resampling.LANCZOS)
                img.save(out / f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="输入视频目录")
    parser.add_argument("--output_folder", required=True, help="输出目录")
    parser.add_argument("--total_frames", type=int, default=100)
    parser.add_argument("--matching_method", choices=["vocab_tree", "exhaustive", "sequential"], default="vocab_tree")
    parser.add_argument("--refine_intrinsics", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_folder)
    images_path = output / "images"

    print("开始提取视频帧...")
    extract_frames_ffmpeg(args.input_folder, images_path, total_frames=args.total_frames)
    
    print("开始相机标定...")
    run_colmap(
        image_dir=images_path,
        colmap_dir=output,
        matching_method=args.matching_method,
        refine_intrinsics=args.refine_intrinsics,
    )

    print("开始去畸变...")
    run_undistort(output)

    print("开始下采样...")
    undistorted_images = output / "undistorted/images"
    downscale_images(undistorted_images)


if __name__ == "__main__":
    main()
