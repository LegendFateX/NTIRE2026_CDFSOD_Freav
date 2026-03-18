import json
import torch
import fire
import os
import glob
from tqdm import tqdm
from collections import defaultdict
from torchvision.ops import nms

def xywh2xyxy(bbox):
    """Detectron2 预测的 [x, y, w, h] 转为 PyTorch NMS 需要的 [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def process_single_file(input_json, output_json, iou_thresh, class_agnostic):
    """处理单个 JSON 文件的核心逻辑"""
    with open(input_json, 'r') as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        # 如果文件为空，直接保存空列表
        with open(output_json, 'w') as f:
            json.dump([], f)
        return 0, 0

    # 按照 image_id 对所有预测结果进行分组
    img_to_preds = defaultdict(list)
    for pred in predictions:
        img_to_preds[pred['image_id']].append(pred)

    new_predictions = []
    
    # 禁用单个文件内部的 tqdm 进度条，避免批量处理时刷屏，改为外层显示进度
    for img_id, img_preds in img_to_preds.items():
        if class_agnostic:
            # 【无类别 NMS】
            boxes = torch.tensor([xywh2xyxy(p['bbox']) for p in img_preds], dtype=torch.float32)
            scores = torch.tensor([p['score'] for p in img_preds], dtype=torch.float32)
            
            keep_idx = nms(boxes, scores, iou_thresh)
            for idx in keep_idx:
                new_predictions.append(img_preds[idx.item()])
                
        else:
            # 【逐类别 NMS】
            cat_to_preds = defaultdict(list)
            for p in img_preds:
                cat_to_preds[p['category_id']].append(p)
                
            for cat_id, cat_preds in cat_to_preds.items():
                if len(cat_preds) == 0:
                    continue
                    
                boxes = torch.tensor([xywh2xyxy(p['bbox']) for p in cat_preds], dtype=torch.float32)
                scores = torch.tensor([p['score'] for p in cat_preds], dtype=torch.float32)
                
                keep_idx = nms(boxes, scores, iou_thresh)
                
                for idx in keep_idx:
                    new_predictions.append(cat_preds[idx.item()])

    with open(output_json, 'w') as f:
        json.dump(new_predictions, f)
        
    return len(predictions), len(new_predictions)

def batch_process_prediction_nms(
    input_dir: str = '/home/lufangxiao/NTIRE2026_CDFSOD/collected_jsons', 
    output_dir: str = '/home/lufangxiao/NTIRE2026_CDFSOD/output_result', 
    iou_thresh: float = 0.35,
    class_agnostic: bool = False
):
    """
    批量对文件夹下的所有预测 JSON 进行 NMS 后处理
    :param input_dir: 包含多个 JSON 文件的输入文件夹
    :param output_dir: NMS 过滤后的 JSON 输出文件夹
    :param iou_thresh: NMS 重叠阈值 (默认 0.5)
    :param class_agnostic: 是否开启跨类别 NMS (默认 False, 即逐类别进行 NMS)
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 查找输入文件夹下所有的 json 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if len(json_files) == 0:
        print(f"No JSON files found in '{input_dir}'.")
        return

    print(f"Found {len(json_files)} JSON files. Starting batch NMS processing (IoU: {iou_thresh})...")

    total_boxes_before = 0
    total_boxes_after = 0

    # 外层进度条，显示处理了多少个文件
    for json_path in tqdm(json_files, desc="Processing JSON files"):
        filename = os.path.basename(json_path)
        out_path = os.path.join(output_dir, filename)
        print(json_path)
        
        before, after = process_single_file(json_path, out_path, iou_thresh, class_agnostic)
        total_boxes_before += before
        total_boxes_after += after

    print("="*50)
    print("Batch Processing Completed!")
    print(f"Total Files Processed   : {len(json_files)}")
    print(f"Total Boxes Before NMS  : {total_boxes_before}")
    print(f"Total Boxes After NMS   : {total_boxes_after}")
    print(f"Total Filtered Boxes    : {total_boxes_before - total_boxes_after}")
    print(f"Saved all results to    : {output_dir}")
    print("="*50)

if __name__ == "__main__":
    fire.Fire(batch_process_prediction_nms)