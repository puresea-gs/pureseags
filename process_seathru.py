import os
import json
import shutil

# 你刚才跑通的三个新场景
scenes = ["IUI3-RedSea", "JapaneseGradens-RedSea", "Panama"]
base_dir = "/path/to/your/dataset"

for scene in scenes:
    print(f"\n========== 正在处理场景: {scene} ==========")
    
    orig_depth_dir = os.path.join(base_dir, scene, "depths")
    ns_dir = os.path.join(base_dir, f"{scene}_nerfstudio")
    ns_depth_dir = os.path.join(ns_dir, "depths")
    json_path = os.path.join(ns_dir, "transforms.json")
    
    # 1. 复制深度图文件夹到 nerfstudio 目录
    if not os.path.exists(ns_depth_dir):
        print("📁 正在拷贝 depths 文件夹...")
        shutil.copytree(orig_depth_dir, ns_depth_dir)
    
    # 2. 将深度图重命名为 frame_xxxxx.png 以对齐原图
    depth_files = sorted([f for f in os.listdir(ns_depth_dir) if f.endswith('.png')])
    if depth_files and not depth_files[0].startswith("frame_"):
        print("🔄 正在重命名深度图...")
        for i, old_name in enumerate(depth_files):
            new_name = f"frame_{i + 1:05d}.png"
            os.rename(os.path.join(ns_depth_dir, old_name), os.path.join(ns_depth_dir, new_name))
            
    # 3. 修改 transforms.json 注入深度约束路径
    print("📝 正在注入 transforms.json...")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    meta["depth_unit_scale_factor"] = 1.0 # 加入尺度因子
    for frame in meta["frames"]:
        img_name = os.path.basename(frame["file_path"])
        frame["depth_file_path"] = f"depths/{img_name}" # 写入对应深度图路径

    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"🎉 {scene} 深度图与 JSON 绑定完成！")

print("\n🚀 所有数据集准备完毕，随时可以开训！")