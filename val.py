"""
YOLO 验证脚本 - 在测试集上评估模型性能
"""

import os
import sys
import argparse
from ultralytics import YOLO


def prepare_data_yaml():
    """根据脚本所在路径自动生成 data.yaml"""
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
    parser = argparse.ArgumentParser(description="YOLO 烟火检测验证")
    parser.add_argument("--model", type=str, default="runs/train/exp/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--device", type=str, default="0", help="设备 (0,1,2,3 或 cpu)")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU 阈值")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="验证集划分")
    args = parser.parse_args()

    # 检查模型
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        sys.exit(1)

    # 自动生成 data.yaml
    data_yaml = prepare_data_yaml()

    print("=" * 60)
    print("YOLO 烟火检测验证")
    print("=" * 60)
    print(f"  模型:    {args.model}")
    print(f"  数据:    {data_yaml}")
    print(f"  划分:    {args.split}")
    print(f"  图片:    {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  设备:    {args.device}")
    print("=" * 60)

    # 加载模型
    model = YOLO(args.model)

    # 执行验证（使用 YOLO 默认保存路径: runs/val/exp）
    results = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
    )

    # 打印关键指标
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    metrics = results.results_dict
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 输出每个类别的 AP
    if hasattr(results, "ap_class_index") and hasattr(results, "ap"):
        print(f"\n  各类别 AP:")
        for i, cls_name in enumerate(["fire", "smoke"]):
            ap50 = results.ap[i][0] if len(results.ap) > i else 0
            ap50_95 = results.ap[i].mean() if len(results.ap) > i else 0
            print(f"    {cls_name}: AP@50={ap50:.4f}, AP@50:95={ap50_95:.4f}")

    print(f"\n结果保存至: runs/val/exp*")
    print("=" * 60)


if __name__ == "__main__":
    main()
