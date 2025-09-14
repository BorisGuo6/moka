#!/usr/bin/env python3
"""
批量运行 MolmoAct 模型处理 RLBench 图像

对每个任务图像运行10次，生成轨迹可视化并保存到对应的子文件夹中。
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
    
    # 查找深度标记
    depth_pattern = r'<DEPTH_START>(.*?)<DEPTH_END>'
    match = re.search(depth_pattern, text)
    
    if not match:
        return None
    
    depth_tokens = match.group(1)
    # 提取所有深度值
    depth_values = re.findall(r'<DEPTH_(\d+)>', depth_tokens)
    return [int(val) for val in depth_values]

def parse_trajectory_tokens(text):
    """解析轨迹标记"""
    import re
    
    # 查找轨迹标记
    traj_pattern = r'The trajectory of the end effector in the first image is \[\[(\d+)'
    match = re.search(traj_pattern, text)
    
    if not match:
        return None
    
    # 尝试提取更多轨迹点
    traj_pattern_full = r'\[\[(\d+.*?)\]\]'
    matches = re.findall(traj_pattern_full, text)
    
    if matches:
        # 解析第一个匹配的轨迹
        traj_str = matches[0]
        # 简单的数字提取
        numbers = re.findall(r'\d+', traj_str)
        if len(numbers) >= 2:
            # 假设是成对的坐标
            points = []
            for i in range(0, len(numbers)-1, 2):
                if i+1 < len(numbers):
                    points.append([int(numbers[i]), int(numbers[i+1])])
            return points
    
    return None

def create_trajectory_overlay(img_array, trajectory_points, output_path):
    """创建轨迹叠加图像"""
    if not trajectory_points or len(trajectory_points) == 0:
        return False
    
    # 创建轨迹叠加
    overlay_img = img_array.copy()
    
    # 获取原始图像尺寸
    img_height, img_width = img_array.shape[:2]
    
    # 假设 MolmoAct 模型内部使用 224x224 分辨率
    model_resolution = 224
    scale_x = img_width / model_resolution
    scale_y = img_height / model_resolution
    
    # 绘制轨迹点
    for i, point in enumerate(trajectory_points):
        if len(point) >= 2:
            # 将模型坐标缩放到原始图像坐标
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)
            
            # 确保坐标在图像范围内
            x = max(0, min(x, img_width-1))
            y = max(0, min(y, img_height-1))
            
            # 根据图像大小调整圆圈大小
            circle_radius = max(4, int(4 * min(scale_x, scale_y)))
            border_radius = max(6, int(6 * min(scale_x, scale_y)))
            
            # 绘制点 (红色圆圈)
            cv2.circle(overlay_img, (x, y), circle_radius, (255, 0, 0), -1)  # 红色填充圆
            cv2.circle(overlay_img, (x, y), border_radius, (0, 0, 0), 1)    # 黑色边框
            
            # 添加点编号
            font_scale = max(0.4, 0.4 * min(scale_x, scale_y))
            cv2.putText(overlay_img, str(i+1), (x-3, y-8), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
            # 绘制连接线
            if i > 0 and len(trajectory_points[i-1]) >= 2:
                prev_x = int(trajectory_points[i-1][0] * scale_x)
                prev_y = int(trajectory_points[i-1][1] * scale_y)
                prev_x = max(0, min(prev_x, img_width-1))
                prev_y = max(0, min(prev_y, img_height-1))
                
                # 根据图像大小调整线条粗细
                line_thickness = max(2, int(2 * min(scale_x, scale_y)))
                cv2.line(overlay_img, (prev_x, prev_y), (x, y), (0, 255, 0), line_thickness)  # 绿色连接线
    
    # 保存轨迹叠加图像
    cv2.imwrite(output_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    return True

def process_single_image(model, processor, image_path, task_instruction, output_dir, run_id):
    """处理单张图像"""
    print(f"  Run {run_id+1}/10: Processing {image_path.name}")
    
    try:
        # 加载图像
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # 构建提示词
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
        
        # 应用聊天模板
        text = processor.apply_chat_template(
            [{"role": "user", "content": [dict(type="text", text=prompt)]}], 
            tokenize=False, 
            add_generation_prompt=True,
        )
        
        # 处理输入
        inputs = processor(
            images=[[img]],
            text=text,
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到正确的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成输出
        import torch
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_ids = model.generate(**inputs, max_new_tokens=256)
        
        # 解码生成的文本
        generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        generated_text = processor.batch_decode(
            generated_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 解析深度和轨迹
        depth_values = parse_depth_tokens(generated_text)
        trajectory_points = parse_trajectory_tokens(generated_text)
        
        # 保存结果
        result = {
            "run_id": run_id + 1,
            "task_instruction": task_instruction,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "depth_values": depth_values,
            "trajectory_points": trajectory_points,
            "generated_text": generated_text,
            "depth_count": len(depth_values) if depth_values else 0,
            "trajectory_count": len(trajectory_points) if trajectory_points else 0
        }
        
        # 保存 JSON 结果
        json_path = output_dir / f"run_{run_id+1:02d}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 创建轨迹叠加图像
        if trajectory_points:
            overlay_path = output_dir / f"run_{run_id+1:02d}_trajectory_overlay.png"
            success = create_trajectory_overlay(img_array, trajectory_points, str(overlay_path))
            result["overlay_created"] = success
        else:
            result["overlay_created"] = False
        
        # 尝试解析动作
        try:
            action = model.parse_action(generated_text, unnorm_key="molmoact")
            result["parsed_action"] = action.tolist() if hasattr(action, 'tolist') else action
        except Exception as e:
            result["parsed_action"] = None
            result["action_parse_error"] = str(e)
        
        print(f"    ✅ Completed: {len(trajectory_points) if trajectory_points else 0} trajectory points")
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {
            "run_id": run_id + 1,
            "task_instruction": task_instruction,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "success": False
        }

def batch_process_rlbench():
    """批量处理 RLBench 图像"""
    print("🚀 Starting batch MolmoAct processing for RLBench images...")
    
    # 任务配置
    task_configs = {
        "CloseBox.jpg": "Close the box.",
        "OpenWineBottle.jpg": "Open the wine bottle by grabbing the cap.",
        "PickUpCup.jpg": "Pick up the red cup.",
        "PlugChargerInPowerSupply.png": "Plug the charger into the power supply.",
        "PutShoesInBox.jpeg": "Put the right shoe in the box.",
        "SlideCabinetOpenAndPlaceCups.png": "Slide the cabinet open and place the cup into the cabinet.",
        "TakeFrameOffHanger.png": "Take the frame off the hanger.",
        "TakePlateOffColoredDishRack.png": "Take the plate off the colored dish rack."
    }
    
    # 设置路径
    rlbench_dir = Path("/home/nus/zjx/moka/rlbench")
    output_base_dir = Path("/home/nus/zjx/moka/scripts/rlbench_results")
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
            attn_implementation="eager"
        )
        
        # 修复注意力实现
        for name, module in model.named_modules():
            if hasattr(module, 'attn_implementation') and module.attn_implementation is None:
                module.attn_implementation = "eager"
        
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # 处理每个任务
    all_results = {}
    
    for image_file, task_instruction in task_configs.items():
        image_path = rlbench_dir / image_file
        
        if not image_path.exists():
            print(f"⚠️ Image not found: {image_path}")
            continue
        
        # 创建任务输出目录
        task_name = image_path.stem
        task_output_dir = output_base_dir / task_name
        task_output_dir.mkdir(exist_ok=True)
        
        print(f"\n📋 Processing task: {task_name}")
        print(f"   Instruction: {task_instruction}")
        print(f"   Output dir: {task_output_dir}")
        
        task_results = []
        
        # 运行10次
        for run_id in range(10):
            result = process_single_image(
                model, processor, image_path, task_instruction, 
                task_output_dir, run_id
            )
            task_results.append(result)
            
            # 短暂延迟避免过热
            time.sleep(1)
        
        all_results[task_name] = task_results
        
        # 保存任务汇总
        summary = {
            "task_name": task_name,
            "task_instruction": task_instruction,
            "image_path": str(image_path),
            "total_runs": len(task_results),
            "successful_runs": len([r for r in task_results if r.get("success", True)]),
            "avg_trajectory_points": np.mean([r.get("trajectory_count", 0) for r in task_results]),
            "avg_depth_values": np.mean([r.get("depth_count", 0) for r in task_results]),
            "results": task_results
        }
        
        summary_path = task_output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Task completed: {summary['successful_runs']}/10 successful runs")
    
    # 保存全局汇总
    global_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(all_results),
        "total_runs": sum(len(results) for results in all_results.values()),
        "successful_runs": sum(
            len([r for r in results if r.get("success", True)]) 
            for results in all_results.values()
        ),
        "tasks": all_results
    }
    
    global_summary_path = output_base_dir / "global_summary.json"
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Batch processing completed!")
    print(f"   Total tasks: {global_summary['total_tasks']}")
    print(f"   Total runs: {global_summary['total_runs']}")
    print(f"   Successful runs: {global_summary['successful_runs']}")
    print(f"   Results saved to: {output_base_dir}")
    
    return True

if __name__ == "__main__":
    success = batch_process_rlbench()
    sys.exit(0 if success else 1)
