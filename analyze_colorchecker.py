"""分析比色板颜色值，对比各实验版本的差异"""
import torch
from torchvision.io import read_image, ImageReadMode
import os, glob, json

SCENE = "Curasao"
IMG_NAME = "MTN_1302.jpg"  # 含有比色板的图片

ABLATION_DIR = "/home/ubuntu/files/LZF/water-splatting0324/ablation_results"
BASELINE_DIR = "/home/ubuntu/files/LZF/water-splatting0324/final_results_ours_0524/images_Curasao/test"

# ===== Step 1: 在 GT 图上找出比色板位置 =====
gt_path = f"{ABLATION_DIR}/images_{SCENE}_round1_dcp/test/gt-rgb/{IMG_NAME}"
gt = read_image(gt_path, ImageReadMode.RGB).float() / 255.0
_, H, W = gt.shape
print(f"GT image: {W}x{H}")

# 比色板通常有24个色块，每个色块有独特的RGB组合
# 扫描寻找连续的高方差区域
# 先在 GT 上计算亮度梯度，找边缘密集区域
from torch.nn.functional import conv2d

gray = gt.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
# Sobel 边缘检测
sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().view(1,1,3,3)
sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().view(1,1,3,3)
edge_x = conv2d(gray, sobel_x, padding=1)
edge_y = conv2d(gray, sobel_y, padding=1)
edge_mag = torch.sqrt(edge_x**2 + edge_y**2).squeeze()

# 找边缘密集区域 → 用滑动窗口统计边缘密度
kernel_size = 50
edge_density = torch.nn.functional.avg_pool2d(edge_mag.unsqueeze(0).unsqueeze(0), kernel_size, stride=25).squeeze()

# 找到边缘密度最高的区域
max_density_idx = edge_density.argmax().item()
density_h, density_w = edge_density.shape
max_y = max_density_idx // density_w
max_x = max_density_idx % density_w

# 映射回原图坐标
y_center = max_y * 25 + kernel_size // 2
x_center = max_x * 25 + kernel_size // 2
print(f"Color checker likely at center: ({x_center}, {y_center})")

# ===== Step 2: 手动定义比色板patches（先打印区域颜色验证）=====
# 以检测到的中心为基准，覆盖24个色块的大致区域
patch_size = 30  # 每个色块大小
rows, cols = 4, 6  # Macbeth ColorChecker 是 4x6
patch_h, patch_w = 40, 40  # 色块间距

# 检查中心区域的实际颜色分布
print(f"\n--- 检查中心区域 ({x_center-150}:{x_center+150}, {y_center-100}:{y_center+100}) ---")
center_crop = gt[:, max(0,y_center-100):min(H,y_center+100), max(0,x_center-150):min(W,x_center+150)]
# 在裁剪区域内按网格采样
ch, cw = center_crop.shape[1:]
for yy in range(0, ch, ch//5):
    for xx in range(0, cw, cw//5):
        rgb = center_crop[:, yy, xx]
        print(f"  rel({xx},{yy}): RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})")

# ===== Step 3: 如果上面区域不对，扫描更多区域 =====
# 也看看右边/下边区域
print(f"\n--- 检查右下区域 ({W-400}:{W-100}, {H-300}:{H-50}) ---")
crop_br = gt[:, H-300:H-50, W-400:W-100]
ch, cw = crop_br.shape[1:]
for yy in range(0, ch, ch//4):
    for xx in range(0, cw, cw//6):
        rgb = crop_br[:, yy, xx]
        print(f"  rel({xx},{yy}): RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})")

print(f"\n--- 检查左下区域 (0:300, {H-300}:{H-50}) ---")
crop_bl = gt[:, H-300:H-50, 0:300]
ch, cw = crop_bl.shape[1:]
for yy in range(0, ch, ch//4):
    for xx in range(0, cw, cw//6):
        rgb = crop_bl[:, yy, xx]
        print(f"  rel({xx},{yy}): RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})")

print(f"\n--- 检查右侧中部 ({W-350}:{W-50}, {H//2-150}:{H//2+150}) ---")
crop_rm = gt[:, H//2-150:H//2+150, W-350:W-50]
ch, cw = crop_rm.shape[1:]
for yy in range(0, ch, ch//4):
    for xx in range(0, cw, cw//6):
        rgb = crop_rm[:, yy, xx]
        print(f"  rel({xx},{yy}): RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})")

print(f"\n--- 检查左侧中部 (0:300, {H//2-150}:{H//2+150}) ---")
crop_lm = gt[:, H//2-150:H//2+150, 0:300]
ch, cw = crop_lm.shape[1:]
for yy in range(0, ch, ch//4):
    for xx in range(0, cw, cw//6):
        rgb = crop_lm[:, yy, xx]
        print(f"  rel({xx},{yy}): RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})")


# ===== Step 4: 多图片分析 =====
# 如果有 colorchecker 在 MTN_1306.jpg 中，也看看
print("\n\n=== MTN_1306.jpg ===")
gt_path2 = f"{ABLATION_DIR}/images_{SCENE}_round1_dcp/test/gt-rgb/MTN_1306.jpg"
gt2 = read_image(gt_path2, ImageReadMode.RGB).float() / 255.0
_, H2, W2 = gt2.shape
print(f"Image size: {W2}x{H2}")
for img_name in ["MTN_1306.jpg", "MTN_1288.jpg"]:
    gt_path = f"{ABLATION_DIR}/images_{SCENE}_round1_dcp/test/gt-rgb/{img_name}"
    gt_tmp = read_image(gt_path, ImageReadMode.RGB).float() / 255.0
    _, h, w = gt_tmp.shape
    print(f"\n--- {img_name} ({w}x{h}) ---")
    # 右下角
    crop = gt_tmp[:, h-300:h-50, w-400:w-100]
    ch, cw = crop.shape[1:]
    for yy in range(0, ch, max(ch//4,1)):
        for xx in range(0, cw, max(cw//6,1)):
            rgb = crop[:, yy, xx]
            print(f"  BR({xx},{yy}): RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})")
