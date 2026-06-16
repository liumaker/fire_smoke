"""
YOLO 训练脚本 (Ultralytics)
使用 yolo_dataset 训练烟火检测模型
"""

import os
from ultralytics import YOLO

# ============ 配置 ============
MODEL = "yolo11n.pt"           # 预训练模型: yolo11n/s/m/l/x
EPOCHS = 100
BATCH = 16
IMGSZ = 640
DEVICE = 0                     # GPU 设备号, CPU 设为 'cpu'
WORKERS = 8
PROJECT = "runs/train"
NAME = "fire_smoke"
RESUME = False                 # 是否从断点恢复
# ==============================


def prepare_data_yaml():
    """根据脚本所在路径自动生成 data.yaml（确保路径正确）"""
    base = os.path.dirname(__file__)
    data_yaml = os.path.join(base, "yolo_dataset", "data.yaml")

    content = f"""train: {os.path.join(base, 'yolo_dataset', 'images', 'train').replace(os.sep, '/')}
val: {os.path.join(base, 'yolo_dataset', 'images', 'val').replace(os.sep, '/')}
test: {os.path.join(base, 'yolo_dataset', 'images', 'test').replace(os.sep, '/')}

nc: 2
names: ["fire", "smoke"]
"""
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(content)
    return data_yaml


def main():
    # 自动生成 data.yaml（路径跟随脚本位置）
    data_yaml = prepare_data_yaml()
    print(f"  数据配置:  {data_yaml}")

    print("=" * 60)
    print("YOLO 烟火检测训练")
    print("=" * 60)
    print(f"  模型:      {MODEL}")
    print(f"  数据:      {data_yaml}")
    print(f"  轮次:      {EPOCHS}")
    print(f"  Batch:     {BATCH}")
    print(f"  图片尺寸:  {IMGSZ}")
    print(f"  设备:      {DEVICE}")
    print("=" * 60)

    # 加载模型
    model = YOLO(MODEL)

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        resume=RESUME,
        amp=True,                     # 混合精度训练
        cache=False,                  # 是否缓存图片到内存
        patience=20,                  # EarlyStopping 耐心值
        seed=42,
    )

    print(f"\n训练完成! 模型保存至: {os.path.join(PROJECT, NAME)}")


if __name__ == "__main__":
    main()
