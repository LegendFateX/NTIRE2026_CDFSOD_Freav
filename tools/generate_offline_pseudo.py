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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

sys.path.append(osp.join(osp.dirname(__file__), ".."))
# 我们不再需要 resize_to_closest_14x，滑窗函数会自动处理
from extract_instance_prototypes import normalize_image

def extract_dense_features(model, image_tensor, scale_factor=2.0, crop_size=504, stride=126, device='cuda:0'):
    """
    带重叠的高分辨率滑动窗口特征提取
    :param scale_factor: 放大倍数，2.0 意味着特征图分辨率翻倍 (步长从 14 降到 7)
    :param crop_size: 滑动窗口大小 (必须是 14 的倍数)
    :param stride: 滑动步长 (重叠度，252 代表 50% 重叠)
    """
    _, H_ori, W_ori = image_tensor.shape
    H_large, W_large = int(H_ori * scale_factor), int(W_ori * scale_factor)
    
    # 保证放大后的尺寸是 14 的倍数
    H_large = math.ceil(H_large / 14) * 14
    W_large = math.ceil(W_large / 14) * 14
    
    img_large = F.interpolate(image_tensor.unsqueeze(0), size=(H_large, W_large), mode='bicubic', align_corners=False)[0]
    
    # Padding 计算，确保滑窗能够完全覆盖边缘
    H_pad, W_pad = H_large, W_large
    if H_pad < crop_size: H_pad = crop_size
    if W_pad < crop_size: W_pad = crop_size
    
    if (H_pad - crop_size) % stride != 0:
        H_pad += stride - ((H_pad - crop_size) % stride)
    if (W_pad - crop_size) % stride != 0:
        W_pad += stride - ((W_pad - crop_size) % stride)
        
    img_padded = F.pad(img_large, (0, W_pad - W_large, 0, H_pad - H_large))
    
    feat_h, feat_w = H_pad // 14, W_pad // 14
    C = None
    global_feat = None
    global_count = torch.zeros((1, feat_h, feat_w), device=device, dtype=torch.float32)
    
    # 开始滑窗提取
    for y in range(0, H_pad - crop_size + 1, stride):
        for x in range(0, W_pad - crop_size + 1, stride):
            crop = img_padded[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
            
            with torch.no_grad():
                r = model.get_intermediate_layers(crop, return_class_token=True, reshape=True)
                feat = r[0][0][0] # [C, crop_size/14, crop_size/14]
            
            if C is None:
                C = feat.shape[0]
                global_feat = torch.zeros((C, feat_h, feat_w), device=device, dtype=torch.float32)
                
            y_f, x_f = y // 14, x // 14
            h_f, w_f = crop_size // 14, crop_size // 14
            
            global_feat[:, y_f:y_f+h_f, x_f:x_f+w_f] += feat
            global_count[:, y_f:y_f+h_f, x_f:x_f+w_f] += 1
            
    # 取均值融合
    global_feat = global_feat / global_count.clamp(min=1.0)
    
    # 裁回大图尺寸
    valid_h, valid_w = H_large // 14, W_large // 14
    return global_feat[:, :valid_h, :valid_w]


@torch.no_grad()
def main(
    train_json: str,                     
    image_root: str,                     
    out_json: str = './train_pseudo.json', 
    model_name: str = 'vitl14', 
    prototypes_pkl: str = '',            
    bg_prototypes_file: str = '/home/lufangxiao/NTIRE2026_CDFSOD/weights/background/background_prototypes.vitl14.pth',        
    out_vis_dir: str = './offline_pseudo_vis', 
    num_images_to_vis: int = 10,         
    vis_all_classes: bool = True,        
    device: str = '0'
):
    if device != 'cpu': device = "cuda:" + str(device)
    os.makedirs(out_vis_dir, exist_ok=True)

    print(f"Loading COCO Annotations from {train_json} ...")
    with open(train_json, 'r') as f: coco_data = json.load(f)

    categories = sorted(coco_data.get('categories', []), key=lambda x: x['id'])
    contig_to_raw = {i: cat['id'] for i, cat in enumerate(categories)}
    raw_to_contig = {cat['id']: i for i, cat in enumerate(categories)}
    raw_to_name = {cat['id']: cat.get('name', str(cat['id'])) for cat in categories}

    max_anno_id = 0
    img_to_annos = {img['id']: [] for img in coco_data['images']}
    for anno in coco_data['annotations']:
        anno['is_pseudo'] = False
        anno['soft_weight'] = 1.0
        img_to_annos[anno['image_id']].append(anno)
        if anno['id'] > max_anno_id: max_anno_id = anno['id']

    print("Loading DINOv2 Model and Prototypes...")
    model = torch.hub.load('/home/lufangxiao/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_' + model_name, source='local')
    model = model.to(device)
    model.eval()

    assert prototypes_pkl != ''
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

    print("Generating Offline Pseudo Labels with Sliding Window...")
    new_annotations = []
    
    for img_idx, img_info in enumerate(tqdm(coco_data['images'])):
        img_id = img_info['id']
        img_path = osp.join(image_root, img_info['file_name'])
        if not osp.exists(img_path): continue
            
        img_np = cv2.imread(img_path)
        if img_np is None: continue
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        H_ori, W_ori = img_rgb.shape[:2]

        image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        
        # =====================================================================
        # 【核心升级】：启用高分辨率滑动窗口特征提取！
        # 放大 2.0 倍提取，相当于 7x7 原始像素生成 1 个特征点，细节爆炸
        # =====================================================================
        patch_tokens = extract_dense_features(
            model, 
            normalize_image(image_tensor), 
            scale_factor=4.0, 
            crop_size=504, 
            stride=252, 
            device=device
        )
        
        C, H_patch, W_patch = patch_tokens.shape
        pt_norm = F.normalize(patch_tokens.reshape(C, -1).permute(1, 0), p=2, dim=-1)

        query_tasks = []
        target_classes = list(range(num_classes)) if vis_all_classes else list(set([a['category_id'] for a in img_to_annos[img_id]]))
        for cls_id in target_classes:
            if cls_id >= num_classes: continue
            query_tasks.append((f"cls_{raw_to_name[contig_to_raw[cls_id]]}", fg_protos_norm[cls_id].unsqueeze(0)))
            
        if bg_protos_norm is not None:
            query_tasks.append(("Background", bg_protos_norm))

        all_sim_maps = []   
        overlay_images = [] 
        task_names = []

        for task_name, protos in query_tasks:
            spatial_sim = torch.matmul(pt_norm, protos.T).max(dim=-1)[0] 
            all_sim_maps.append(spatial_sim)
            
            if img_idx < num_images_to_vis:
                sim_map = np.clip(spatial_sim.view(H_patch, W_patch).cpu().numpy(), 0, 1) 
                sim_map_resized = cv2.resize(sim_map, (W_ori, H_ori), interpolation=cv2.INTER_CUBIC)
                heatmap = np.uint8(255 * sim_map_resized)
                heatmap_colored = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)
                overlay_images.append(overlay)
                task_names.append(task_name)

        temperature = 0.05
        sims = torch.matmul(pt_norm, all_protos.T) 
        probs = F.softmax(sims / temperature, dim=1)
        
        top2_probs, _ = torch.topk(probs, k=2, dim=1)
        margin_1d = top2_probs[:, 0] - top2_probs[:, 1]
        margin_map = cv2.resize(margin_1d.view(H_patch, W_patch).cpu().numpy(), (W_ori, H_ori), interpolation=cv2.INTER_CUBIC)

        seg_idx_1d = sims.argmax(dim=1)
        seg_resized = cv2.resize(seg_idx_1d.view(H_patch, W_patch).cpu().numpy().astype(np.uint8), (W_ori, H_ori), interpolation=cv2.INTER_NEAREST)

        visited_mask = np.zeros((H_ori, W_ori), dtype=bool)
        for anno in img_to_annos[img_id]:
            x, y, w, h = map(int, anno['bbox'])
            c_contig = raw_to_contig.get(anno['category_id'], None)
            if c_contig is not None:
                box_mask = np.zeros((H_ori, W_ori), dtype=bool)
                box_mask[y:y+h, x:x+w] = True
                visited_mask |= (box_mask & (seg_resized == c_contig))

        # =====================================================================
        # 移除形态学腐蚀，直接基于高分辨率特征进行图斑打分与提框
        # =====================================================================
        cmap = plt.get_cmap('tab20')
        colors = (cmap(np.linspace(0, 1, len(query_tasks)))[:, :3] * 255).astype(np.uint8)
        core_colored = np.zeros((H_ori, W_ori, 3), dtype=np.uint8)

        candidates = []
        min_area_threshold = (W_ori * H_ori) * 0.0005 
        conf_threshold = 0.4 # 高置信度过滤

        for c_i in range(len(query_tasks)):
            if query_tasks[c_i][0] == "Background": continue
            
            class_mask = (seg_resized == c_i) & (~visited_mask)
            
            # 【移除腐蚀】仅保留强置信度过滤
            core_mask = class_mask & (margin_map > conf_threshold)
            
            if img_idx < num_images_to_vis:
                core_colored[core_mask] = colors[c_i]

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(core_mask.astype(np.uint8), connectivity=8)
            
            for j in range(1, num_labels):
                x, y, w, h, area = stats[j]
                if area > min_area_threshold: 
                    comp_mask = (labels == j)
                    score = float(margin_map[comp_mask].mean())
                    
                    candidates.append({
                        'class_id': target_classes[c_i],
                        'score': score,
                        'bbox': [x, y, w, h],
                        'mask': comp_mask 
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        current_pseudo_boxes = []
        for cand in candidates:
            comp_mask = cand['mask']
            if (comp_mask & visited_mask).sum() / comp_mask.sum() < 0.05: 
                visited_mask |= comp_mask
                
                max_anno_id += 1
                pseudo_anno = {
                    "id": max_anno_id,
                    "image_id": img_id,
                    "category_id": contig_to_raw[cand['class_id']],
                    "bbox": [int(v) for v in cand['bbox']], 
                    "area": int(cand['bbox'][2] * cand['bbox'][3]),
                    "iscrowd": 0,
                    "is_pseudo": True,
                    "soft_weight": round(cand['score'], 4)
                }
                new_annotations.append(pseudo_anno)
                current_pseudo_boxes.append(pseudo_anno)

        # =====================================================================
        # 可视化 Debug 网格 (移除了腐蚀图，保留核心图)
        # =====================================================================
        if img_idx < num_images_to_vis:
            img_boxes = img_rgb.copy()
            for anno in img_to_annos[img_id]:
                x, y, w, h = map(int, anno['bbox'])
                cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            for anno in current_pseudo_boxes:
                x, y, w, h = map(int, anno['bbox'])
                cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_boxes, f"P:{anno['soft_weight']:.2f}", (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            margin_colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * margin_map), cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
            
            seg_colored = np.zeros((H_ori, W_ori, 3), dtype=np.uint8)
            for i in range(len(query_tasks)): seg_colored[seg_resized == i] = colors[i]
            
            seg_overlay = cv2.addWeighted(img_rgb, 0.4, seg_colored, 0.6, 0)
            core_overlay = cv2.addWeighted(img_rgb, 0.4, core_colored, 0.6, 0)

            total_plots = 6 + len(overlay_images)
            cols = 4 
            rows = math.ceil(total_plots / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()
            
            axes[0].imshow(img_boxes); axes[0].set_title("GT & Pseudo Boxes", fontsize=12, fontweight='bold'); axes[0].axis('off')
            axes[1].imshow(seg_overlay); axes[1].set_title("Coarse Mask (High Res)", fontsize=12, fontweight='bold'); axes[1].axis('off')
            axes[2].imshow(margin_colored); axes[2].set_title("Margin (High Res)", fontsize=12, fontweight='bold'); axes[2].axis('off')
            axes[3].imshow(core_overlay); axes[3].set_title(f"Core Mask (> {conf_threshold})", fontsize=12, fontweight='bold'); axes[3].axis('off')
            
            axes[4].axis('off'); axes[4].set_title("Legend", fontsize=14, fontweight='bold')
            legend_elements = [Patch(facecolor=colors[j]/255, edgecolor='black', label=task_names[j]) for j in range(len(task_names))]
            axes[4].legend(handles=legend_elements, loc='center', fontsize=10, ncol=2 if len(task_names)>10 else 1)

            axes[5].axis('off'); axes[5].set_title("Margin Scale", fontsize=14, fontweight='bold')
            cax = axes[5].inset_axes([0.1, 0.4, 0.8, 0.2]) 
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.set_ticks([0.0, 0.5, 1.0])

            for i in range(len(overlay_images)):
                ax_idx = i + 6
                axes[ax_idx].imshow(overlay_images[i])
                axes[ax_idx].set_title(f"Heatmap: {task_names[i]}", fontsize=12)
                axes[ax_idx].axis('off')
                
            for i in range(total_plots, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            save_path = osp.join(out_vis_dir, f"{img_id}_mining_grid_hires.png")
            plt.savefig(save_path, dpi=200) 
            plt.close()

    print(f"Generated {len(new_annotations)} pseudo annotations.")
    coco_data['annotations'].extend(new_annotations)
    with open(out_json, 'w') as f: json.dump(coco_data, f)
    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)