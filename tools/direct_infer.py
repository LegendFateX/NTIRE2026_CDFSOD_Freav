import os
import sys
import math
import json
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import fire
from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from extract_instance_prototypes import normalize_image

def extract_dense_features_unfold(
    model, 
    image_tensor, 
    scale_factor=2.0, 
    crop_size=504, 
    stride=504, # 步长等于窗大小，代表零重叠，速度极致
    device='cuda:0'
):
    """
    基于 PyTorch 原生 unfold 和 fold 的零循环、全并行特征提取
    """
    _, H_ori, W_ori = image_tensor.shape
    H_large, W_large = int(H_ori * scale_factor), int(W_ori * scale_factor)
    
    # 保证放大后的尺寸是 14 的倍数 (DINOv2 的步长)
    H_large = math.ceil(H_large / 14) * 14
    W_large = math.ceil(W_large / 14) * 14
    
    # 1. 插值放大
    img_large = F.interpolate(
        image_tensor.unsqueeze(0), 
        size=(H_large, W_large), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 2. Padding 计算，确保能被滑窗完整切分
    H_pad, W_pad = H_large, W_large
    if H_pad < crop_size: H_pad = crop_size
    if W_pad < crop_size: W_pad = crop_size
    
    if (H_pad - crop_size) % stride != 0:
        H_pad += stride - ((H_pad - crop_size) % stride)
    if (W_pad - crop_size) % stride != 0:
        W_pad += stride - ((W_pad - crop_size) % stride)
        
    img_padded = F.pad(img_large, (0, W_pad - W_large, 0, H_pad - H_large)) # [1, 3, H_pad, W_pad]
    
    # =====================================================================
    # 绝杀：使用 unfold 直接在空间维度提取所有 Patches
    # =====================================================================
    # unfold(dim, size, step) -> [1, 3, n_h, n_w, crop_size, crop_size]
    patches = img_padded.unfold(2, crop_size, stride).unfold(3, crop_size, stride)
    n_h, n_w = patches.shape[2], patches.shape[3]
    num_patches = n_h * n_w
    
    # 变换为批量维度: [num_patches, 3, crop_size, crop_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(num_patches, 3, crop_size, crop_size)
    patches = patches.to(device)
    
    # 3. DINOv2 全并行一步推理 (直接吞下所有的 Patches)
    with torch.no_grad():
        r = model.get_intermediate_layers(patches, return_class_token=True, reshape=True)
        feats = r[0][0] # 形状: [num_patches, C, h_feat, w_feat]
        
    C = feats.shape[1]
    h_feat, w_feat = crop_size // 14, crop_size // 14
    
    # 映射回全局特征图的尺寸和步长
    feat_h_pad, feat_w_pad = H_pad // 14, W_pad // 14
    stride_feat = stride // 14
    
    # =====================================================================
    # 绝杀：使用 F.fold 将乱序特征一键拼接回大图 (自动处理相加重叠)
    # =====================================================================
    # fold 的输入要求形状为: [1, C * h_feat * w_feat, num_patches]
    feats_folded_input = feats.reshape(num_patches, C * h_feat * w_feat).transpose(0, 1).unsqueeze(0)
    
    global_feat = F.fold(
        feats_folded_input,
        output_size=(feat_h_pad, feat_w_pad),
        kernel_size=(h_feat, w_feat),
        stride=(stride_feat, stride_feat)
    ) # 形状: [1, C, feat_h_pad, feat_w_pad]
    
    # 如果步长存在重叠 (stride < crop_size)，需要计算重叠次数以求平均
    if stride < crop_size:
        ones_input = torch.ones_like(feats_folded_input)
        overlap_count = F.fold(
            ones_input,
            output_size=(feat_h_pad, feat_w_pad),
            kernel_size=(h_feat, w_feat),
            stride=(stride_feat, stride_feat)
        )
        global_feat = global_feat / overlap_count.clamp(min=1.0)
        
    global_feat = global_feat.squeeze(0) # [C, feat_h_pad, feat_w_pad]
    
    # 4. 裁切掉为了 unfold 而增加的 Padding，还原真实尺寸
    valid_h, valid_w = H_large // 14, W_large // 14
    return global_feat[:, :valid_h, :valid_w]

@torch.no_grad()
def main(
    test_json: str,                      
    image_root: str,                     
    out_json: str = './coco_instances_results_hires.json', 
    model_name: str = 'vitl14', 
    prototypes_pkl: str = '',            
    bg_prototypes_file: str = '/home/lufangxiao/NTIRE2026_CDFSOD/weights/background/background_prototypes.vitl14.pth',        
    conf_threshold: float = 0.3,         
    scale_factor: float = 2.0,           
    crop_size: int = 504,                
    stride: int = 504,                   # 默认设为 504 零重叠！
    device: str = '0'
):
    if device != 'cpu': device = "cuda:" + str(device)

    print(f"Loading Test Annotations from {test_json} ...")
    with open(test_json, 'r') as f: 
        coco_data = json.load(f)

    categories = sorted(coco_data.get('categories', []), key=lambda x: x['id'])
    contig_to_raw = {i: cat['id'] for i, cat in enumerate(categories)}

    print("Loading DINOv2 Model and Prototypes...")
    model = torch.hub.load('/home/lufangxiao/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_' + model_name, source='local')
    model = model.to(device)
    model.eval()

    assert prototypes_pkl != '', "You must provide prototypes_pkl!"
    dct = torch.load(prototypes_pkl, map_location='cpu')
    prototypes = dct['prototypes'].to(device) 
    num_classes = len(prototypes)

    fg_protos_mean = prototypes.mean(dim=1) if prototypes.dim() == 3 else prototypes
    fg_protos_norm = F.normalize(fg_protos_mean, p=2, dim=-1)

    bg_protos_norm = None
    if bg_prototypes_file != '':
        bg_protos = torch.load(bg_prototypes_file, map_location='cpu')
        if isinstance(bg_protos, dict): bg_protos = bg_protos['prototypes']
        if len(bg_protos.shape) == 3: bg_protos = bg_protos.flatten(0, 1)
        bg_protos_norm = F.normalize(bg_protos.to(device), p=2, dim=-1)
    
    all_protos = torch.cat([fg_protos_norm, bg_protos_norm], dim=0) if bg_protos_norm is not None else fg_protos_norm

    print(f"Running Unfold-Fold High-Res Inference (Scale: {scale_factor}x, Stride: {stride})...")
    
    predictions = []
    
    for img_info in tqdm(coco_data['images']):
        img_id = img_info['id']
        img_path = osp.join(image_root, img_info['file_name'])
        if not osp.exists(img_path): 
            continue
            
        img_np = cv2.imread(img_path)
        if img_np is None: continue
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        H_ori, W_ori = img_rgb.shape[:2]

        image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        
        # 调用极速版 Unfold-Fold 特征提取！
        patch_tokens = extract_dense_features_unfold(
            model, 
            normalize_image(image_tensor), 
            scale_factor=scale_factor, 
            crop_size=crop_size, 
            stride=stride, 
            device=device
        )
        
        C, H_patch, W_patch = patch_tokens.shape
        pt_norm = F.normalize(patch_tokens.reshape(C, -1).permute(1, 0), p=2, dim=-1)

        temperature = 0.05
        sims = torch.matmul(pt_norm, all_protos.T) 
        probs = F.softmax(sims / temperature, dim=1)
        
        top2_probs, _ = torch.topk(probs, k=2, dim=1)
        margin_1d = top2_probs[:, 0] - top2_probs[:, 1]
        margin_map = cv2.resize(margin_1d.view(H_patch, W_patch).cpu().numpy(), (W_ori, H_ori), interpolation=cv2.INTER_CUBIC)

        seg_idx_1d = sims.argmax(dim=1)
        seg_resized = cv2.resize(seg_idx_1d.view(H_patch, W_patch).cpu().numpy().astype(np.uint8), (W_ori, H_ori), interpolation=cv2.INTER_NEAREST)

        candidates = []
        min_area_threshold = (W_ori * H_ori) * 0.0005 

        for c_i in range(num_classes):
            core_mask = (seg_resized == c_i) & (margin_map > conf_threshold)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(core_mask.astype(np.uint8), connectivity=8)
            
            for j in range(1, num_labels):
                x, y, w, h, area = stats[j]
                if area > min_area_threshold: 
                    comp_mask = (labels == j)
                    score = float(margin_map[comp_mask].mean())
                    
                    candidates.append({
                        'image_id': img_id,
                        'category_id': contig_to_raw[c_i],
                        'score': score,
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'mask': comp_mask 
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        visited_mask = np.zeros((H_ori, W_ori), dtype=bool)
        for cand in candidates:
            comp_mask = cand['mask']
            if (comp_mask & visited_mask).sum() / comp_mask.sum() < 0.05: 
                visited_mask |= comp_mask
                
                predictions.append({
                    "image_id": cand['image_id'],
                    "category_id": cand['category_id'],
                    "bbox": cand['bbox'], 
                    "score": round(cand['score'], 4)
                })

    print(f"Inference Completed. Total bounding boxes predicted: {len(predictions)}")
    print(f"Saving predictions to {out_json} ...")
    with open(out_json, 'w') as f: 
        json.dump(predictions, f)
    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)