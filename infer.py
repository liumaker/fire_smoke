"""
YOLO 推理脚本 - 对单张图片、视频或目录批量推理
"""

import os
import sys
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO 烟火检测推理")
    parser.add_argument("source", type=str, help="输入源: 图片/视频路径 或 目录")
    parser.add_argument("--model", type=str, default="runs/train/exp/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备 (0,1,2,3 或 cpu)")
    parser.add_argument("--show", action="store_true", help="显示结果")
    parser.add_argument("--save-txt", action="store_true", help="保存标签文件 (.txt)")
    args = parser.parse_args()

    # 检查模型
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        sys.exit(1)

    print("=" * 60)
    print("YOLO 烟火检测推理")
    print("=" * 60)
    print(f"  模型:    {args.model}")
    print(f"  来源:    {args.source}")
    print(f"  置信度:  {args.conf}")
    print(f"  IoU:     {args.iou}")
    print("=" * 60)

    # 加载模型
    model = YOLO(args.model)

    # 执行推理（使用 YOLO 默认保存路径: runs/detect/exp）
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        show=args.show,
        save_txt=args.save_txt,
    )

    print(f"\n推理完成! 共处理 {len(results)} 张图片")
    print(f"结果保存至: runs/detect/exp*")


if __name__ == "__main__":
    main()
