import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="统计人工标注结果")
    parser.add_argument("--input", required=True, help="人工标注后的 CSV 文件")
    parser.add_argument("--output-prefix", default="annotation_stats", help="输出统计文件前缀")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")

    print("总样本数:", len(df))
    print()

    overall = df["human_label"].value_counts().reset_index()
    overall.columns = ["human_label", "count"]

    by_road = pd.crosstab(df["road_id"], df["human_label"])
    by_road["total"] = by_road.sum(axis=1)

    by_heading = pd.crosstab(df["heading"], df["human_label"])
    by_heading["total"] = by_heading.sum(axis=1)

    by_road_heading = pd.crosstab(
        [df["road_id"], df["heading"]],
        df["human_label"]
    )
    by_road_heading["total"] = by_road_heading.sum(axis=1)

    print("总体统计:")
    print(overall)
    print()

    print("按 road_id 统计:")
    print(by_road)
    print()

    print("按 heading 统计:")
    print(by_heading)
    print()

    overall.to_csv(f"{args.output_prefix}_overall.csv", index=False, encoding="utf-8-sig")
    by_road.to_csv(f"{args.output_prefix}_by_road.csv", encoding="utf-8-sig")
    by_heading.to_csv(f"{args.output_prefix}_by_heading.csv", encoding="utf-8-sig")
    by_road_heading.to_csv(f"{args.output_prefix}_by_road_heading.csv", encoding="utf-8-sig")

    print("统计完成，已输出统计 CSV。")


if __name__ == "__main__":
    main()