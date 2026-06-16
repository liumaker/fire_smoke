"""
VOC2007 -> YOLO 格式转换脚本
将 1w张预处理后融合数据集/VOC2007 转换为 YOLO 训练格式

类别:
  - fire (0)
  - smoke (1)
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET

# ============ 配置 ============
VOC_ROOT = r"c:\Users\elitedatai\Desktop\work\fire_smoke\1w张预处理后融合数据集\VOC2007"
YOLO_ROOT = r"c:\Users\elitedatai\Desktop\work\fire_smoke\yolo_dataset"

CLASSES = ["fire", "smoke"]  # fire=0, smoke=1

# 划分比例 (训练:验证:测试)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)
# =============================

ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
JPEGIMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")


def parse_voc_annotation(xml_path):
    """解析单个 VOC XML 文件, 返回 (image_filename, width, height, bboxes)
    
    bboxes: [(class_id, xmin, ymin, xmax, ymax), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 图片文件名
    filename = root.find("filename").text

    # 图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    bboxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            print(f"  [警告] 未知类别 '{name}' 在 {xml_path}, 跳过")
            continue
        class_id = CLASSES.index(name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        bboxes.append((class_id, xmin, ymin, xmax, ymax))

    return filename, width, height, bboxes


def convert_to_yolo_bbox(width, height, bbox):
    """将 VOC (xmin, ymin, xmax, ymax) 转换为 YOLO (x_center, y_center, w, h) 归一化坐标"""
    class_id, xmin, ymin, xmax, ymax = bbox

    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height

    return class_id, x_center, y_center, bbox_width, bbox_height


def main():
    print("=" * 60)
    print("VOC2007 -> YOLO 格式转换")
    print("=" * 60)

    # 1. 收集所有包含目标的 XML 文件
    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
    print(f"\n[1/4] 找到 {len(xml_files)} 个 XML 标注文件")

    # 2. 解析所有有效标注
    valid_samples = []
    no_object_count = 0
    no_image_count = 0

    for xml_file in sorted(xml_files):
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)

        try:
            filename, width, height, bboxes = parse_voc_annotation(xml_path)
        except Exception as e:
            print(f"  [错误] 解析 {xml_file} 失败: {e}")
            continue

        if not bboxes:
            no_object_count += 1
            continue

        # 检查对应图片是否存在
        img_path = os.path.join(JPEGIMAGES_DIR, filename)
        if not os.path.exists(img_path):
            # 尝试 jpg / png 等多种后缀
            base_name = os.path.splitext(filename)[0]
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                alt_path = os.path.join(JPEGIMAGES_DIR, base_name + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    found = True
                    break
            if not found:
                no_image_count += 1
                print(f"  [警告] {xml_file} 对应的图片 '{filename}' 不存在, 跳过")
                continue

        # 获取真实图片尺寸 (以防 XML 中尺寸不准确)
        # 尽量使用入站尺寸

        valid_samples.append({
            "xml_path": xml_path,
            "img_path": img_path,
            "img_name": os.path.basename(img_path),
            "base_name": os.path.splitext(os.path.basename(img_path))[0],
            "width": width,
            "height": height,
            "bboxes": bboxes,
        })

    print(f"         其中 {no_object_count} 个没有目标, {no_image_count} 个缺少图片")
    print(f"         有效样本: {len(valid_samples)} 个")

    if not valid_samples:
        print("\n[错误] 没有有效的标注样本, 退出!")
        return

    # 3. 划分训练/验证/测试集
    random.shuffle(valid_samples)
    total = len(valid_samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_samples = valid_samples[:train_end]
    val_samples = valid_samples[train_end:val_end]
    test_samples = valid_samples[val_end:]

    print(f"\n[2/4] 数据集划分:")
    print(f"     训练集: {len(train_samples)} 张")
    print(f"     验证集: {len(val_samples)} 张")
    print(f"     测试集: {len(test_samples)} 张")

    # 4. 创建 YOLO 目录结构
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    print(f"\n[3/4] 创建 YOLO 目录结构...")
    for split_name in splits:
        os.makedirs(os.path.join(YOLO_ROOT, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(YOLO_ROOT, "labels", split_name), exist_ok=True)
    print(f"     目录: {YOLO_ROOT}")

    # 5. 复制图片并生成 YOLO 标签文件
    print(f"\n[4/4] 开始转换...")
    total_converted = 0
    for split_name, samples in splits.items():
        for sample in samples:
            base_name = sample["base_name"]

            # 复制图片
            dst_img = os.path.join(YOLO_ROOT, "images", split_name, f"{base_name}.jpg")
            try:
                shutil.copy2(sample["img_path"], dst_img)
            except Exception as e:
                print(f"  [错误] 复制图片失败 {sample['img_path']}: {e}")
                continue

            # 生成 YOLO 标签文件
            label_path = os.path.join(YOLO_ROOT, "labels", split_name, f"{base_name}.txt")
            with open(label_path, "w") as f:
                for bbox in sample["bboxes"]:
                    class_id, x_center, y_center, bw, bh = convert_to_yolo_bbox(
                        sample["width"], sample["height"], bbox
                    )
                    # 裁剪到 [0, 1] 范围以防越界
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    bw = max(0, min(1, bw))
                    bh = max(0, min(1, bh))
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

            total_converted += 1

            if total_converted % 500 == 0:
                print(f"     已转换 {total_converted}/{len(valid_samples)} ...")

    print(f"\n{'=' * 60}")
    print(f"转换完成!")
    print(f"  输出目录: {YOLO_ROOT}")
    print(f"  总样本数: {total_converted}")
    print(f"  类别: {', '.join(f'{i}: {c}' for i, c in enumerate(CLASSES))}")
    print(f"{'=' * 60}")

    # 输出 YOLO 训练配置
    print(f"\n数据集配置文件 data.yaml 内容:")
    print(f"  {YOLO_ROOT}\\data.yaml")
    print(f"\n{'=' * 60}")
    print("train: " + os.path.join(YOLO_ROOT, "images", "train").replace("\\", "/"))
    print("val: " + os.path.join(YOLO_ROOT, "images", "val").replace("\\", "/"))
    print("test: " + os.path.join(YOLO_ROOT, "images", "test").replace("\\", "/"))
    print(f"\nnc: {len(CLASSES)}")
    print(f"names: {CLASSES}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
