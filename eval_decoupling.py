import os
import glob
import cv2
import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import warnings
import csv

# 屏蔽烦人的底层警告
warnings.filterwarnings("ignore")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 3DGS 论文同款评测标准
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    # 原始的绝对纯净 GT 数据目录
    gt_base_dir = "/home/ubuntu/files/LZF/Mip-NeRF360/garden/images"
    experiments = ["foggy", "underwater"]

    print("==================================================")
    print("🏆 开始进行终极解耦评估 (rgb_clear vs Ground Truth)")
    print("==================================================")

    results = []

    for exp in experiments:
        search_pattern = f"final_simulated_results/images_{exp}/rgb_clear/**/*.*"
        pred_paths = [p for p in glob.glob(search_pattern, recursive=True) if p.endswith(('.png', '.jpg', '.JPG'))]
        
        total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
        count = 0
        
        for pred_path in pred_paths:
            basename = os.path.basename(pred_path)
            name_no_ext = os.path.splitext(basename)[0]
            
            # 去找原图
            gt_path = os.path.join(gt_base_dir, f"{name_no_ext}.JPG")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_base_dir, f"{name_no_ext}.jpg")
                if not os.path.exists(gt_path):
                    continue
            
            # 读取预测的 rgb_clear
            pred_img = cv2.imread(pred_path)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # 读取纯净 GT 
            gt_img = cv2.imread(gt_path)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # 缩小 GT 对齐分辨率
            h, w = pred_img.shape[:2]
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)
            
            # 转为 Tensor 
            pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0).to(device)
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 累计分数
            total_psnr += psnr_metric(pred_tensor, gt_tensor).item()
            total_ssim += ssim_metric(pred_tensor, gt_tensor).item()
            total_lpips += lpips_metric(pred_tensor, gt_tensor).item()
            count += 1
            
        if count > 0:
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            avg_lpips = total_lpips / count
            print(f"| {exp.capitalize():<10} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | (共比对 {count} 张)")
            results.append([exp.capitalize(), f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_lpips:.4f}"])
        else:
            print(f"| {exp.capitalize():<10} | 未找到匹配图片，请检查是否成功渲染！")
            
    print("==================================================")
    
    # 🔥 保存到 CSV
    if results:
        csv_path = "final_simulated_results/decoupling_metrics_summary.csv"
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Scene", "PSNR", "SSIM", "LPIPS"])
            writer.writerows(results)
        print(f"✅ 解耦评估数据已保存至 CSV: {csv_path}")

if __name__ == "__main__":
    main()