import argparse
import csv
import shutil
from pathlib import Path


def complete_reused_images(input_csv, output_csv=None, image_dir=None, overwrite=False):
    input_csv = Path(input_csv)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_completed.csv")
    else:
        output_csv = Path(output_csv)

    rows = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("CSV 文件为空或没有表头")

        required = {"sample_id", "image_path"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError("CSV 缺少必要字段: " + ",".join(sorted(missing)))

        for row in reader:
            rows.append(row)

    copied_count = 0
    skipped_count = 0
    missing_count = 0

    for row in rows:
        sample_id = row["sample_id"].strip()
        old_image_path = Path(row["image_path"].strip())

        if image_dir is not None:
            target_image_path = Path(image_dir) / f"{sample_id}.jpg"
        else:
            target_image_path = old_image_path.parent / f"{sample_id}.jpg"

        # 如果 CSV 里已经是正常路径，不需要复制
        if old_image_path.as_posix() == target_image_path.as_posix():
            skipped_count += 1
            continue

        # 源文件不存在，记录一下，不中断整个流程
        if not old_image_path.exists():
            print(f"源图片不存在，跳过: {old_image_path}")
            missing_count += 1
            row["image_path"] = target_image_path.as_posix()
            continue

        target_image_path.parent.mkdir(parents=True, exist_ok=True)

        if target_image_path.exists() and not overwrite:
            print(f"目标图片已存在，跳过复制: {target_image_path}")
        else:
            shutil.copy2(old_image_path, target_image_path)
            print(f"复制复用图片: {old_image_path} -> {target_image_path}")
            copied_count += 1

        # 无论是否已经存在，都把 CSV 改成正常路径
        row["image_path"] = target_image_path.as_posix()

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("处理完成")
    print(f"输入 CSV: {input_csv}")
    print(f"输出 CSV: {output_csv}")
    print(f"复制图片数: {copied_count}")
    print(f"已是正常路径/跳过数: {skipped_count}")
    print(f"源图片缺失数: {missing_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="补全复用图片文件，使每个 sample_id 都有对应命名的图片")
    parser.add_argument("--input", required=True, help="输入采样点 CSV")
    parser.add_argument("--output", default=None, help="输出修正后的 CSV，默认在原文件名后加 _completed")
    parser.add_argument("--image-dir", default=None, help="图片目录；不填则使用原 image_path 所在目录")
    parser.add_argument("--overwrite", action="store_true", help="如果目标图片已存在，是否覆盖")
    return parser.parse_args()


def main():
    args = parse_args()
    complete_reused_images(
        input_csv=args.input,
        output_csv=args.output,
        image_dir=args.image_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()