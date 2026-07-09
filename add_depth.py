import json
import os

# 你的 Nerfstudio 数据集路径
base_dir = "/path/to/your/nerfstudio/dataset"
json_path = os.path.join(base_dir, "transforms.json")

with open(json_path, 'r') as f:
    meta = json.load(f)

# 确保全局设置了深度图的缩放因子 (Depth Anything 视差没绝对物理尺度，设 1.0 即可，我们在代码里用 lstsq 对齐了)
meta["depth_unit_scale_factor"] = 1.0

# 遍历所有帧，注入 depth_file_path
for frame in meta["frames"]:
    # 提取纯文件名，例如 frame_00001.png
    img_name = os.path.basename(frame["file_path"])
    
    # 假设你的深度图也是 .png 结尾，且名字和 rgb 一样
    # 如果你的深度图是 .npy，这里改成 .replace(".png", ".npy")
    depth_name = img_name 
    
    # 写入深度图的相对路径
    frame["depth_file_path"] = f"depths/{depth_name}"

# 将修改后的字典写回 json
with open(json_path, 'w') as f:
    json.dump(meta, f, indent=4)

print("✅ 成功！深度图路径已注入 transforms.json！")