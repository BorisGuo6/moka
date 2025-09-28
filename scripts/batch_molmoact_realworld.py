#!/usr/bin/env python3
"""
批量运行 MolmoAct 模型处理 realworld 文件夹里的所有图片

遍历 /home/nus/zjx/moka/realworld 下的所有子文件夹与图片：
- 文字指令为图片父文件夹名称（例如 .../Move the tomoto into the pan/rgb_init.png -> "Move the tomoto into the pan"）
- 对每张图片运行10次，保存轨迹可视化、解析结果与汇总
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
import time
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_depth_tokens(text):
    """解析深度标记"""
    import re

    depth_pattern = r'<DEPTH_START>(.*?)<DEPTH_END>'
    match = re.search(depth_pattern, text)
    if not match:
        return None

    depth_tokens = match.group(1)
    depth_values = re.findall(r'<DEPTH_(\d+)>', depth_tokens)
    return [int(val) for val in depth_values]


def parse_trajectory_tokens(text):
    """解析轨迹标记"""
    import re

    traj_pattern = r'The trajectory of the end effector in the first image is \[\[(\d+)'
    match = re.search(traj_pattern, text)
    if not match:
        return None

    traj_pattern_full = r'\[\[(\d+.*?)\]\]'
    matches = re.findall(traj_pattern_full, text)
    if matches:
        traj_str = matches[0]
        numbers = re.findall(r'\d+', traj_str)
        if len(numbers) >= 2:
            points = []
            for i in range(0, len(numbers) - 1, 2):
                if i + 1 < len(numbers):
                    points.append([int(numbers[i]), int(numbers[i + 1])])
            return points

    return None


def create_trajectory_overlay(img_array, trajectory_points, output_path):
    """创建轨迹叠加图像"""
    if not trajectory_points or len(trajectory_points) == 0:
        return False

    overlay_img = img_array.copy()
    img_height, img_width = img_array.shape[:2]

    model_resolution = 224
    scale_x = img_width / model_resolution
    scale_y = img_height / model_resolution

    for i, point in enumerate(trajectory_points):
        if len(point) >= 2:
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))

            circle_radius = max(4, int(4 * min(scale_x, scale_y)))
            border_radius = max(6, int(6 * min(scale_x, scale_y)))

            cv2.circle(overlay_img, (x, y), circle_radius, (255, 0, 0), -1)
            cv2.circle(overlay_img, (x, y), border_radius, (0, 0, 0), 1)

            font_scale = max(0.4, 0.4 * min(scale_x, scale_y))
            cv2.putText(
                overlay_img,
                str(i + 1),
                (x - 3, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )

            if i > 0 and len(trajectory_points[i - 1]) >= 2:
                prev_x = int(trajectory_points[i - 1][0] * scale_x)
                prev_y = int(trajectory_points[i - 1][1] * scale_y)
                prev_x = max(0, min(prev_x, img_width - 1))
                prev_y = max(0, min(prev_y, img_height - 1))
                line_thickness = max(2, int(2 * min(scale_x, scale_y)))
                cv2.line(overlay_img, (prev_x, prev_y), (x, y), (0, 255, 0), line_thickness)

    cv2.imwrite(output_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    return True


def process_single_image(model, processor, image_path, task_instruction, output_dir, run_id):
    """处理单张图像"""
    print(f"  Run {run_id + 1}/10: Processing {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        prompt = (
            f"The task is {task_instruction}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {task_instruction}, "
            "let's think through it step by step. "
            "First, what is the depth map for the first image? "
            "Second, what is the trajectory of the end effector in the first image? "
            "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
            "along with other images from different camera views as additional information, "
            "what is the action that the robot should take?"
        )

        text = processor.apply_chat_template(
            [{"role": "user", "content": [dict(type="text", text=prompt)]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            images=[[img]],
            text=text,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        import torch
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_ids = model.generate(**inputs, max_new_tokens=256)

        generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        generated_text = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        depth_values = parse_depth_tokens(generated_text)
        trajectory_points = parse_trajectory_tokens(generated_text)

        result = {
            "run_id": run_id + 1,
            "task_instruction": task_instruction,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "depth_values": depth_values,
            "trajectory_points": trajectory_points,
            "generated_text": generated_text,
            "depth_count": len(depth_values) if depth_values else 0,
            "trajectory_count": len(trajectory_points) if trajectory_points else 0,
        }

        json_path = output_dir / f"run_{run_id + 1:02d}_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if trajectory_points:
            overlay_path = output_dir / f"run_{run_id + 1:02d}_trajectory_overlay.png"
            success = create_trajectory_overlay(img_array, trajectory_points, str(overlay_path))
            result["overlay_created"] = success
        else:
            result["overlay_created"] = False

        try:
            action = model.parse_action(generated_text, unnorm_key="molmoact")
            result["parsed_action"] = action.tolist() if hasattr(action, "tolist") else action
        except Exception as e:
            result["parsed_action"] = None
            result["action_parse_error"] = str(e)

        print(
            f"    ✅ Completed: {len(trajectory_points) if trajectory_points else 0} trajectory points"
        )
        return result

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {
            "run_id": run_id + 1,
            "task_instruction": task_instruction,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "success": False,
        }


def batch_process_realworld():
    """批量处理 realworld 文件夹中的所有图片"""
    print("🚀 Starting batch MolmoAct processing for realworld images...")

    # 根路径与输出路径
    realworld_dir = Path("/home/nus/zjx/moka/realworld")
    output_base_dir = Path("/home/nus/zjx/moka/scripts/realworld_results")
    output_base_dir.mkdir(exist_ok=True)

    # 加载模型
    print("📦 Loading MolmoAct model...")
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch

        ckpt = "allenai/MolmoAct-7B-D-0812"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            padding_side="left",
        )

        model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager",
        )

        for name, module in model.named_modules():
            if hasattr(module, "attn_implementation") and module.attn_implementation is None:
                module.attn_implementation = "eager"

        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # 收集所有图片
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = [p for p in realworld_dir.rglob("*") if p.suffix.lower() in img_exts]
    if len(image_paths) == 0:
        print(f"⚠️ No images found under: {realworld_dir}")
        return False

    all_results = {}

    for image_path in sorted(image_paths):
        task_instruction = image_path.parent.name
        task_name = task_instruction

        # 针对同一任务下可能多张图片，按图片名分子目录
        per_image_dir = output_base_dir / task_name / image_path.stem
        per_image_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📋 Processing task: {task_name}")
        print(f"   Image: {image_path}")
        print(f"   Output dir: {per_image_dir}")

        image_results = []
        for run_id in range(10):
            result = process_single_image(
                model, processor, image_path, task_instruction, per_image_dir, run_id
            )
            image_results.append(result)
            time.sleep(1)

        # 保存该图片的汇总
        summary = {
            "task_name": task_name,
            "task_instruction": task_instruction,
            "image_path": str(image_path),
            "total_runs": len(image_results),
            "successful_runs": len([r for r in image_results if r.get("success", True)]),
            "avg_trajectory_points": float(
                np.mean([r.get("trajectory_count", 0) for r in image_results])
            ),
            "avg_depth_values": float(
                np.mean([r.get("depth_count", 0) for r in image_results])
            ),
            "results": image_results,
        }

        with open(per_image_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 汇总至全局
        all_results.setdefault(task_name, {})[image_path.name] = image_results

        print(
            f"   ✅ Image completed: {summary['successful_runs']}/{summary['total_runs']} successful runs"
        )

    # 保存全局汇总
    global_summary = {
        "timestamp": datetime.now().isoformat(),
        "root": str(realworld_dir),
        "total_tasks": len(all_results),
        "total_images": sum(len(v) for v in all_results.values()),
        "tasks": all_results,
    }

    # 重新计算 successful_runs（上面写法会错误嵌套），这里显式计算
    successful_runs = 0
    total_runs = 0
    for task_images in all_results.values():
        for img_results in task_images.values():
            total_runs += len(img_results)
            successful_runs += len([r for r in img_results if r.get("success", True)])
    global_summary["total_runs"] = total_runs
    global_summary["successful_runs"] = successful_runs

    global_summary_path = output_base_dir / "global_summary.json"
    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print("\n🎉 Batch processing completed!")
    print(f"   Total tasks: {global_summary['total_tasks']}")
    print(f"   Total images: {global_summary['total_images']}")
    print(f"   Total runs: {global_summary['total_runs']}")
    print(f"   Successful runs: {global_summary['successful_runs']}")
    print(f"   Results saved to: {output_base_dir}")

    return True


if __name__ == "__main__":
    success = batch_process_realworld()
    sys.exit(0 if success else 1)


