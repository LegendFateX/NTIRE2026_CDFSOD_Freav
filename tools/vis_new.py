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

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from extract_instance_prototypes import normalize_image

def extract_dense_features(
    model, 
    image_tensor, 
    scale_factor=2.0, 
    crop_size=518, 
    stride=504, 
    device='cuda:0'
):
    """
    基于 PyTorch 原生 unfold 和 fold 的零循环、全并行特征提取
    """
    image_tensor = image_tensor.to(device)
    
    _, H_ori, W_ori = image_tensor.shape
    H_large, W_large = int(H_ori * scale_factor), int(W_ori * scale_factor)
    
    H_large = math.ceil(H_large / 14) * 14
    W_large = math.ceil(W_large / 14) * 14
    
    img_large = F.interpolate(
        image_tensor.unsqueeze(0), 
        size=(H_large, W_large), 
        mode='bicubic', 
        align_corners=False
    )
    
    H_pad, W_pad = H_large, W_large
    if H_pad < crop_size: H_pad = crop_size
    if W_pad < crop_size: W_pad = crop_size
    
    if (H_pad - crop_size) % stride != 0:
        H_pad += stride - ((H_pad - crop_size) % stride)
    if (W_pad - crop_size) % stride != 0:
        W_pad += stride - ((W_pad - crop_size) % stride)
        
    img_padded = F.pad(img_large, (0, W_pad - W_large, 0, H_pad - H_large))
    
    patches = img_padded.unfold(2, crop_size, stride).unfold(3, crop_size, stride)
    n_h, n_w = patches.shape[2], patches.shape[3]
    num_patches = n_h * n_w
    
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(num_patches, 3, crop_size, crop_size).to(device)
    
    with torch.no_grad():
        r = model.get_intermediate_layers(patches, return_class_token=True, reshape=True)
        feats = r[0][0] 
        
    C = feats.shape[1]
    h_feat, w_feat = crop_size // 14, crop_size // 14
    
    feat_h_pad, feat_w_pad = H_pad // 14, W_pad // 14
    stride_feat = stride // 14
    
    feats_folded_input = feats.reshape(num_patches, C * h_feat * w_feat).transpose(0, 1).unsqueeze(0)
    
    global_feat = F.fold(
        feats_folded_input,
        output_size=(feat_h_pad, feat_w_pad),
        kernel_size=(h_feat, w_feat),
        stride=(stride_feat, stride_feat)
    )
    
    if stride < crop_size:
        ones_input = torch.ones_like(feats_folded_input)
        overlap_count = F.fold(
            ones_input,
            output_size=(feat_h_pad, feat_w_pad),
            kernel_size=(h_feat, w_feat),
            stride=(stride_feat, stride_feat)
        )
        global_feat = global_feat / overlap_count.clamp(min=1.0)
        
    global_feat = global_feat.squeeze(0) 
    
    valid_h, valid_w = H_large // 14, W_large // 14
    return global_feat[:, :valid_h, :valid_w]


@torch.no_grad()
def main(
    test_json: str,                      
    image_root: str,                     
    out_json: str = './coco_instances_results.json', 
    model_name: str = 'vitl14', 
    prototypes_pkl: str = '',            
    bg_prototypes_file: str = '/home/lufangxiao/NTIRE2026_CDFSOD/weights/background/background_prototypes.vitl14.pth',        
    out_vis_dir: str = './test_infer_vis', 
    num_images_to_vis: int = 20,         
    vis_all_classes: bool = True,
    top_k: int = 3,                      
    scale_factor: float = 1.0,           
    conf_threshold: float = 0.3,         
    device: str = '0'
):
    print(scale_factor)
    if device != 'cpu': device = "cuda:" + str(device)
    os.makedirs(out_vis_dir, exist_ok=True)

    print(f"Loading Test Annotations from {test_json} ...")
    with open(test_json, 'r') as f: 
        coco_data = json.load(f)

    categories = sorted(coco_data.get('categories', []), key=lambda x: x['id'])
    contig_to_raw = {i: cat['id'] for i, cat in enumerate(categories)}
    raw_to_name = {cat['id']: cat.get('name', str(cat['id'])) for cat in categories}

    # 尝试加载 GT，如果测试集没有标注也不报错
    img_to_annos = {img['id']: [] for img in coco_data['images']}
    if 'annotations' in coco_data:
        for anno in coco_data['annotations']:
            img_to_annos[anno['image_id']].append(anno)

    print("Loading DINOv2 Model and Prototypes...")
    model = torch.hub.load('/home/lufangxiao/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_' + model_name, source='local')
    model = model.to(device)
    model.eval()

    assert prototypes_pkl != '', "You must provide prototypes_pkl!"
    dct = torch.load(prototypes_pkl, map_location='cpu')
    prototypes = dct['prototypes'].to(device) 
    
    if prototypes.dim() == 2:
        prototypes = prototypes.unsqueeze(1)
        
    num_classes, num_shots, feat_dim = prototypes.shape
    
    fg_protos_norm = F.normalize(prototypes, p=2, dim=-1) 
    fg_protos_flatten = fg_protos_norm.view(num_classes * num_shots, feat_dim) 

    bg_protos_norm = None
    num_bg_tokens = 0
    if bg_prototypes_file != '':
        bg_protos = torch.load(bg_prototypes_file, map_location='cpu')
        if isinstance(bg_protos, dict): bg_protos = bg_protos['prototypes']
        if len(bg_protos.shape) == 3: bg_protos = bg_protos.flatten(0, 1)
        bg_protos_norm = F.normalize(bg_protos.to(device), p=2, dim=-1)
        num_bg_tokens = bg_protos_norm.shape[0]
    
    all_protos = torch.cat([fg_protos_flatten, bg_protos_norm], dim=0) if bg_protos_norm is not None else fg_protos_flatten

    print(f"Running Inference & Visualization (Scale: {scale_factor}x, Top-{top_k} KNN)...")
    
    predictions = []
    
    for img_idx, img_info in enumerate(tqdm(coco_data['images'])):
        img_id = img_info['id']
        img_path = osp.join(image_root, img_info['file_name'])
        if not osp.exists(img_path): 
            continue
            
        img_np = cv2.imread(img_path)
        if img_np is None: continue
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        H_ori, W_ori = img_rgb.shape[:2]

        image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        
        patch_tokens = extract_dense_features(
            model, 
            normalize_image(image_tensor), 
            scale_factor=scale_factor, 
            crop_size=518, 
            stride=504, 
            device=device
        )
        
        C, H_patch, W_patch = patch_tokens.shape
        pt_norm = F.normalize(patch_tokens.reshape(C, -1).permute(1, 0), p=2, dim=-1)

        # =====================================================================
        # 提取可视化所需的各类 Heatmap
        # =====================================================================
        query_tasks = []
        target_classes = list(range(num_classes)) if vis_all_classes else list(set([a['category_id'] for a in img_to_annos[img_id]]))
        for cls_id in target_classes:
            if cls_id >= num_classes: continue
            query_tasks.append((f"cls_{raw_to_name[contig_to_raw[cls_id]]}", fg_protos_norm[cls_id]))
            
        if bg_protos_norm is not None:
            query_tasks.append(("Background", bg_protos_norm))

        overlay_images = [] 
        task_names = []

        for task_name, protos in query_tasks:
            sims_shots = torch.matmul(pt_norm, protos.T) 
            k_shots = min(top_k, sims_shots.shape[-1])
            spatial_sim = sims_shots.topk(k_shots, dim=-1)[0].mean(dim=-1) 
            
            if img_idx < num_images_to_vis:
                sim_map = np.clip(spatial_sim.view(H_patch, W_patch).cpu().numpy(), 0, 1) 
                sim_map_resized = cv2.resize(sim_map, (W_ori, H_ori), interpolation=cv2.INTER_CUBIC)
                heatmap = np.uint8(255 * sim_map_resized)
                heatmap_colored = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)
                overlay_images.append(overlay)
                task_names.append(task_name)

        # =====================================================================
        # 全局分类与 Softmax 计算
        # =====================================================================
        sims_all = torch.matmul(pt_norm, all_protos.T) 
        
        fg_sims_all = sims_all[:, :num_classes * num_shots].view(-1, num_classes, num_shots)
        k_fg = min(top_k, num_shots)
        fg_topk_sims = fg_sims_all.topk(k_fg, dim=-1)[0].mean(dim=-1) 
        
        softmax_input_list = [fg_topk_sims]
        
        if bg_protos_norm is not None:
            bg_sims_all = sims_all[:, num_classes * num_shots:]
            k_bg = min(top_k, num_bg_tokens)
            bg_topk_sim = bg_sims_all.topk(k_bg, dim=-1)[0].mean(dim=-1) 
            softmax_input_list.append(bg_topk_sim.unsqueeze(1))
            
        probs_input = torch.cat(softmax_input_list, dim=1) 

        temperature = 0.05
        probs = F.softmax(probs_input / temperature, dim=1)
        
        top2_probs, _ = torch.topk(probs, k=2, dim=1)
        margin_1d = top2_probs[:, 0] - top2_probs[:, 1]
        margin_map = cv2.resize(margin_1d.view(H_patch, W_patch).cpu().numpy(), (W_ori, H_ori), interpolation=cv2.INTER_CUBIC)

        seg_idx_1d = probs_input.argmax(dim=1)
        seg_resized = cv2.resize(seg_idx_1d.view(H_patch, W_patch).cpu().numpy().astype(np.uint8), (W_ori, H_ori), interpolation=cv2.INTER_NEAREST)

        # =====================================================================
        # 图斑提取与 COCO 格式包装
        # =====================================================================
        cmap = plt.get_cmap('tab20')
        colors = (cmap(np.linspace(0, 1, len(query_tasks)))[:, :3] * 255).astype(np.uint8)
        core_colored = np.zeros((H_ori, W_ori, 3), dtype=np.uint8)

        candidates = []
        min_area_threshold = (W_ori * H_ori) * 0.0005 

        for c_i in range(num_classes):
            core_mask = (seg_resized == c_i) & (margin_map > conf_threshold)
            
            if img_idx < num_images_to_vis:
                core_colored[core_mask] = colors[c_i]

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
        current_pseudo_boxes = []
        
        for cand in candidates:
            comp_mask = cand['mask']
            if (comp_mask & visited_mask).sum() / comp_mask.sum() < 0.05: 
                visited_mask |= comp_mask
                current_pseudo_boxes.append(cand)
                
                predictions.append({
                    "image_id": cand['image_id'],
                    "category_id": cand['category_id'],
                    "bbox": cand['bbox'], 
                    "score": round(cand['score'], 4)
                })

        # =====================================================================
        # 可视化 Debug 网格生成
        # =====================================================================
        if img_idx < num_images_to_vis:
            img_boxes = img_rgb.copy()
            
            # 如果测试集含有 GT 则画出绿框
            for anno in img_to_annos[img_id]:
                x, y, w, h = map(int, anno['bbox'])
                cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 画出生成的推理框 (红框)
            for cand in current_pseudo_boxes:
                x, y, w, h = map(int, cand['bbox'])
                cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_boxes, f"P:{cand['score']:.2f}", (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            margin_colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * margin_map), cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
            
            seg_colored = np.zeros((H_ori, W_ori, 3), dtype=np.uint8)
            for i in range(len(query_tasks)): seg_colored[seg_resized == i] = colors[i]
            
            seg_overlay = cv2.addWeighted(img_rgb, 0.4, seg_colored, 0.6, 0)
            core_overlay = cv2.addWeighted(img_rgb, 0.4, core_colored, 0.6, 0)

            # 1. 为当前图片创建独立的子文件夹
            individual_dir = osp.join(out_vis_dir, f"{img_id}_parts")
            os.makedirs(individual_dir, exist_ok=True)
            
            # 保存预测框图 (注意：cv2.imwrite 需要转回 BGR)
            cv2.imwrite(osp.join(individual_dir, "prediction_boxes.png"), cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR))

            cv2.imwrite(osp.join(individual_dir, "margin_map.png"), margin_colored)

            cv2.imwrite(osp.join(individual_dir, "segmentation_overlay.png"), cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))

            # 5. 循环保存每一个类别的热力图 (Heatmaps)
            for i, (task_name, _) in enumerate(query_tasks):
                # 这里的 overlay_images[i] 已经是叠加了原图的 RGB 图像
                heatmap_img = overlay_images[i]
                # 清洗文件名，防止非法字符
                safe_name = "".join([c if c.isalnum() else "_" for c in task_name])
                cv2.imwrite(osp.join(individual_dir, f"heatmap_{safe_name}.png"), cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

            total_plots = 6 + len(overlay_images)
            cols = 4 
            rows = math.ceil(total_plots / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()
            
            axes[0].imshow(img_boxes); axes[0].set_title("GT(Green) & Preds(Red)", fontsize=12, fontweight='bold'); axes[0].axis('off')
            axes[1].imshow(seg_overlay); axes[1].set_title(f"Coarse Mask (Top-{top_k} KNN)", fontsize=12, fontweight='bold'); axes[1].axis('off')
            axes[2].imshow(margin_colored); axes[2].set_title("Margin Map", fontsize=12, fontweight='bold'); axes[2].axis('off')
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
            save_path = osp.join(out_vis_dir, f"{img_id}_mining_grid_infer.png")
            plt.savefig(save_path, dpi=200) 
            plt.close()

    print(f"Inference Completed. Total bounding boxes predicted: {len(predictions)}")
    print(f"Saving COCO predictions to {out_json} ...")
    
    with open(out_json, 'w') as f: 
        json.dump(predictions, f)
        
    print(f"Done! Check out {out_vis_dir} for visualization grids.")

if __name__ == "__main__":
    fire.Fire(main)