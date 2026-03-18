import json
import torch
import fire
import os
from tqdm import tqdm
from collections import defaultdict

def xywh2xyxy(bbox):
    """COCO [x, y, w, h] 转 [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def confluence_nms(boxes, scores, md_thresh=0.6):
    """
    Confluence NMS (NMS-C): 替代传统 IoU NMS
    使用归一化曼哈顿距离来衡量重叠度
    
    :param boxes: Tensor [N, 4] (x1, y1, x2, y2)
    :param scores: Tensor [N] 置信度得分
    :param md_thresh: 曼哈顿距离阈值
    :return: 保留的框的索引
    """
    keep = []
    # 按分数从高到低排序 (NMS-C 依然依赖得分来选择最优框)
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        if idxs.numel() == 1:
            keep.append(idxs[0].item())
            break

        current_idx = idxs[0]
        keep.append(current_idx.item())

        current_box = boxes[current_idx]
        rest_boxes = boxes[idxs[1:]]

        # 1. 计算曼哈顿距离 (Manhattan Distance)
        # 对应坐标点相减的绝对值之和
        dist_x1 = torch.abs(current_box[0] - rest_boxes[:, 0])
        dist_y1 = torch.abs(current_box[1] - rest_boxes[:, 1])
        dist_x2 = torch.abs(current_box[2] - rest_boxes[:, 2])
        dist_y2 = torch.abs(current_box[3] - rest_boxes[:, 3])
        
        manhattan_dist = dist_x1 + dist_y1 + dist_x2 + dist_y2

        # 2. 尺寸归一化 (Normalization)
        # 防止大物体和小物体的绝对距离尺度不同，使用两个框的平均宽高进行归一化
        current_w = current_box[2] - current_box[0]
        current_h = current_box[3] - current_box[1]
        
        rest_w = rest_boxes[:, 2] - rest_boxes[:, 0]
        rest_h = rest_boxes[:, 3] - rest_boxes[:, 1]
        
        norm_factor = ((current_w + rest_w) / 2.0) + ((current_h + rest_h) / 2.0)

        # 3. 计算归一化邻近度 (Normalized Proximity)
        # 这个值越接近 0，说明两个框在空间上越贴合
        proximity = manhattan_dist / torch.clamp(norm_factor, min=1e-6)

        # 4. 【核心制裁逻辑】：归一化曼哈顿距离小于阈值，判定为同一物体，予以剔除！
        # 保留那些距离“大于”阈值的框进入下一轮
        keep_condition = proximity > md_thresh

        idxs = idxs[1:][keep_condition]

    return keep

def process_confluence_nms(
    input_json: str = './coco_instances_results.json', 
    output_json: str = './coco_instances_results_confluence.json', 
    md_thresh: float = 0.6
):
    """
    对模型预测的 JSON 进行 Confluence NMS 后处理
    """
    if not os.path.exists(input_json):
        print(f"Error: Could not find {input_json}")
        return

    print(f"Loading predictions from {input_json} ...")
    with open(input_json, 'r') as f:
        predictions = json.load(f)

    # 按 image_id 进行分组
    img_to_preds = defaultdict(list)
    for pred in predictions:
        img_to_preds[pred['image_id']].append(pred)

    new_predictions = []
    total_original = len(predictions)
    
    print(f"Applying Confluence NMS (MD Threshold: {md_thresh}) ...")
    
    for img_id, img_preds in tqdm(img_to_preds.items(), desc="Processing Images"):
        # 按 category_id 进一步分组 (标准逐类别过滤)
        cat_to_preds = defaultdict(list)
        for p in img_preds:
            cat_to_preds[p['category_id']].append(p)
            
        for cat_id, cat_preds in cat_to_preds.items():
            if len(cat_preds) == 0:
                continue
                
            # 转换为 Tensor 加速计算
            boxes = torch.tensor([xywh2xyxy(p['bbox']) for p in cat_preds], dtype=torch.float32)
            scores = torch.tensor([p['score'] for p in cat_preds], dtype=torch.float32)
            
            # 调用 Confluence NMS
            keep_idx = confluence_nms(boxes, scores, md_thresh)
            
            for idx in keep_idx:
                new_predictions.append(cat_preds[idx])

    print("\n" + "="*40)
    print(f"Total Predictions Before Confluence : {total_original}")
    print(f"Total Predictions After Confluence  : {len(new_predictions)}")
    print(f"Filtered out redundant boxes        : {total_original - len(new_predictions)}")
    print("="*40 + "\n")

    print(f"Saving filtered predictions to {output_json} ...")
    with open(output_json, 'w') as f:
        json.dump(new_predictions, f)
    print("Done!")

if __name__ == "__main__":
    fire.Fire(process_confluence_nms)