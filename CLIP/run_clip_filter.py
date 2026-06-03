import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


POSITIVE_PROMPTS = [
    "a photo of a visible electric vehicle charging pile",
    "a photo of a visible EV charging post",
    "a photo of a standalone EV charger pedestal",
    "a photo of a white EV charging cabinet",
    "a photo of an EV charger with a charging cable",
    "a photo of an EV charger with a plug connector",
    "a photo of a charging gun hanging on a charging post",
    "a photo of multiple EV charging cabinets in a row",
    "a close or medium distance photo of EV charging equipment",
    "a photo of a charger post with a cable and connector",
    "a photo of a small white charging pile for electric cars",
    "a photo of electric vehicle charging equipment clearly visible",
]

SUSPECTED_PROMPTS = [
    "a photo of possible EV charging posts",
    "a photo of possible electric vehicle charging equipment",
    "a photo of small cabinet-like devices that may be EV chargers",
    "a distant view of possible EV charging piles",
    "a partially occluded EV charger",
    "a blurry photo of possible charging cabinets",
]

NEGATIVE_PROMPTS = [
    "a photo without EV charging piles",
    "a photo without electric vehicle charging equipment",
    "a photo of ordinary electrical boxes that are not EV chargers",
    "a photo of street lamps and signs without EV chargers",
    "a photo of utility cabinets without charging cables",
    "a photo of parking spaces without EV charging posts",
    "a photo with no charging guns and no charging cables",
    "a photo of an outdoor parking lot without EV chargers",
    "a photo of roadside parking spaces without charging equipment",
    "a photo of a shopping mall parking lot without EV charging piles",
    "a photo of an office building parking area without EV chargers",
    "a street view image with no electric vehicle charging station",
]

INVALID_PROMPTS = [
    "a street view photo from an elevated road",
    "a photo of a highway or overpass road far from the target area",
    "a wrong street view direction",
    "a blurry or invalid street view image",
    "a corrupted or unreadable image",
]

HARD_NEGATIVE_PROMPTS = [
    "a photo of a street sign pole",
    "a photo of a traffic sign pole",
    "a photo of a street light pole",
    "a photo of a metal pole without charging cables",
    "a photo of a bollard or parking post",
    "a photo of an ordinary electrical box",
    "a photo of a utility cabinet without charging plugs",
    "a photo of a transformer box",
    "a photo of a roadside control cabinet",
    "a photo of a parking gate machine",
    "a photo of a guard booth",
    "a photo of a fence post or railing post",
    "a photo of a sign board or information board",
    "a photo of roadside equipment that is not an EV charger",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Use CLIP to roughly filter street-view images.")
    parser.add_argument("--input", default=str(script_dir / "sample_points_annotated.csv"), help="Input annotated CSV")
    parser.add_argument("--output", default=str(script_dir / "clip_results.csv"), help="Output CSV with CLIP scores")
    parser.add_argument("--stats-output", default="", help="Output CSV with summary statistics")
    parser.add_argument("--model", default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", help="HuggingFace CLIP model name or local path")
    parser.add_argument("--batch-size", type=int, default=16, help="Image batch size")
    parser.add_argument("--margin-threshold", type=float, default=-0.02, help="keep_score - strongest negative/invalid score threshold")
    parser.add_argument("--min-keep-score", type=float, default=0.23, help="Minimum positive/suspected score for keep")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows; 0 means all rows")
    parser.add_argument("--crop-grid", default="3x2", help="Crop grid as COLSxROWS, for example 3x2")
    parser.add_argument("--crop-overlap", type=float, default=0.15, help="Overlap ratio between adjacent crops")
    parser.add_argument("--hard-negative-weight", type=float, default=0.50, help="Penalty weight for hard-negative prompts")
    parser.add_argument("--final-threshold", type=float, default=0.115, help="Minimum final score for keep")
    return parser.parse_args()


def resolve_image_path(csv_path, image_path):
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return csv_path.parent / path


def normalize_features(features):
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    elif hasattr(features, "image_embeds"):
        features = features.image_embeds
    elif hasattr(features, "text_embeds"):
        features = features.text_embeds
    return features / features.norm(dim=-1, keepdim=True)


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def parse_crop_grid(value):
    cols_text, rows_text = value.lower().split("x", 1)
    cols = int(cols_text)
    rows = int(rows_text)
    if cols <= 0 or rows <= 0:
        raise ValueError("--crop-grid must use positive integers, like 3x2")
    return cols, rows


def make_crops(image, cols, rows, overlap):
    width, height = image.size
    crop_w = width / cols
    crop_h = height / rows
    pad_w = crop_w * overlap / 2
    pad_h = crop_h * overlap / 2
    crops = []

    for row in range(rows):
        for col in range(cols):
            left = max(0, int(round(col * crop_w - pad_w)))
            top = max(0, int(round(row * crop_h - pad_h)))
            right = min(width, int(round((col + 1) * crop_w + pad_w)))
            bottom = min(height, int(round((row + 1) * crop_h + pad_h)))
            crop_id = f"r{row}_c{col}"
            crops.append((crop_id, image.crop((left, top, right, bottom))))

    return crops


def classify_row(
    keep_score,
    negative_score,
    invalid_score,
    final_score,
    margin_threshold,
    min_keep_score,
    final_threshold,
):
    margin = keep_score - negative_score

    # Coarse filtering prioritizes recall: only drop images clearly unlike EV chargers.
    if keep_score >= min_keep_score and final_score >= final_threshold and margin >= margin_threshold:
        return "keep", margin
    if keep_score >= min_keep_score and margin >= margin_threshold and final_score >= final_threshold - 0.02:
        return "review", margin
    if final_score >= final_threshold:
        return "review", margin
    if margin >= margin_threshold and final_score >= final_threshold - 0.03:
        return "review", margin
    if invalid_score > keep_score and invalid_score > negative_score:
        return "drop_invalid_like", margin
    return "drop", margin


def make_stats_rows(result_df, args, prompts):
    stats = []

    def add(section, metric, value, human_label="", clip_pred="", sample_id=""):
        stats.append(
            {
                "section": section,
                "metric": metric,
                "human_label": human_label,
                "clip_pred": clip_pred,
                "sample_id": sample_id,
                "value": value,
            }
        )

    add("run", "rows", len(result_df))
    add("run", "model", args.model)
    add("run", "margin_threshold", args.margin_threshold)
    add("run", "min_keep_score", args.min_keep_score)
    add("run", "hard_negative_weight", args.hard_negative_weight)
    add("run", "final_threshold", args.final_threshold)
    add("run", "prompt_count", len(prompts))

    for pred, count in result_df["clip_pred"].value_counts(dropna=False).items():
        add("clip_pred_count", "count", count, clip_pred=str(pred))

    if "human_label" in result_df.columns:
        for label, count in result_df["human_label"].value_counts(dropna=False).items():
            add("human_label_count", "count", count, human_label=str(label))

        cross = pd.crosstab(result_df["human_label"], result_df["clip_pred"], dropna=False)
        for human_label in cross.index:
            for clip_pred in cross.columns:
                add(
                    "human_label_by_clip_pred",
                    "count",
                    int(cross.loc[human_label, clip_pred]),
                    str(human_label),
                    str(clip_pred),
                )

        risky = result_df[
            result_df["human_label"].isin(["positive", "suspected"])
            & result_df["clip_pred"].astype(str).str.startswith("drop")
        ]
        add("quality", "risky_drops_positive_or_suspected", len(risky))

    dropped = result_df[result_df["clip_pred"].astype(str).str.startswith("drop")]
    add("quality", "dropped_rows", len(dropped))
    for _, row in dropped.iterrows():
        add(
            "dropped_rows",
            "clip_margin",
            row.get("clip_margin", ""),
            str(row.get("human_label", "")),
            str(row.get("clip_pred", "")),
            str(row.get("sample_id", "")),
        )

    invalid_images = result_df[result_df["clip_pred"].astype(str) == "invalid_image"]
    add("quality", "invalid_image_rows", len(invalid_images))
    for _, row in invalid_images.iterrows():
        add(
            "invalid_image_rows",
            "clip_error",
            row.get("clip_error", ""),
            str(row.get("human_label", "")),
            str(row.get("clip_pred", "")),
            str(row.get("sample_id", "")),
        )

    return stats


def write_stats(stats_path, result_df, args, prompts):
    rows = make_stats_rows(result_df, args, prompts)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "metric", "human_label", "clip_pred", "sample_id", "value"]
    with stats_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output) if args.stats_output else output_path.with_name(output_path.stem + "_stats.csv")
    base_dir = input_path.parent

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "image_path" not in df.columns:
        raise ValueError("Input CSV must contain image_path column")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {args.model}")
    print(f"Device: {device}")

    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = CLIPModel.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        use_safetensors=True,
    ).to(device)
    model.eval()

    prompts = POSITIVE_PROMPTS + SUSPECTED_PROMPTS + NEGATIVE_PROMPTS + INVALID_PROMPTS + HARD_NEGATIVE_PROMPTS
    pos_slice = slice(0, len(POSITIVE_PROMPTS))
    sus_slice = slice(len(POSITIVE_PROMPTS), len(POSITIVE_PROMPTS) + len(SUSPECTED_PROMPTS))
    neg_start = len(POSITIVE_PROMPTS) + len(SUSPECTED_PROMPTS)
    neg_slice = slice(neg_start, neg_start + len(NEGATIVE_PROMPTS))
    inv_start = neg_start + len(NEGATIVE_PROMPTS)
    inv_slice = slice(inv_start, inv_start + len(INVALID_PROMPTS))
    hard_start = inv_start + len(INVALID_PROMPTS)
    hard_slice = slice(hard_start, hard_start + len(HARD_NEGATIVE_PROMPTS))

    with torch.no_grad():
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        text_features = normalize_features(model.get_text_features(**text_inputs))

    if args.limit > 0:
        df = df.head(args.limit).copy()

    rows = df.to_dict("records")
    results = []
    valid_batch = []
    valid_items = []
    crop_cols, crop_rows = parse_crop_grid(args.crop_grid)
    row_scores = {}

    def flush_batch():
        if not valid_batch:
            return

        with torch.no_grad():
            image_inputs = processor(images=valid_batch, return_tensors="pt").to(device)
            image_features = normalize_features(model.get_image_features(**image_inputs))
            sims = image_features @ text_features.T

        for local_idx, item in enumerate(valid_items):
            sim = sims[local_idx]
            positive_score = sim[pos_slice].max().item()
            suspected_score = sim[sus_slice].max().item()
            keep_score = max(positive_score, suspected_score)
            negative_score = sim[neg_slice].max().item()
            invalid_score = sim[inv_slice].max().item()
            hard_negative_score = sim[hard_slice].max().item()
            final_score = keep_score - args.hard_negative_weight * hard_negative_score
            row_idx = item["row_idx"]
            bucket = row_scores.setdefault(row_idx, {"crops": []})
            score = {
                "positive": positive_score,
                "suspected": suspected_score,
                "keep": keep_score,
                "negative": negative_score,
                "invalid": invalid_score,
                "hard_negative": hard_negative_score,
                "final": final_score,
                "crop_id": item["crop_id"],
            }
            if item["kind"] == "global":
                bucket["global"] = score
            else:
                bucket["crops"].append(score)

        valid_batch.clear()
        valid_items.clear()

    for idx, row in enumerate(rows):
        results.append(dict(row))
        image_file = resolve_image_path(base_dir, row["image_path"])
        try:
            image = load_image(image_file)
        except Exception as exc:
            results[idx].update(
                {
                    "clip_positive_score": "",
                    "clip_suspected_score": "",
                    "clip_keep_score": "",
                    "clip_negative_score": "",
                    "clip_invalid_score": "",
                    "clip_margin": "",
                    "clip_global_keep_score": "",
                    "clip_crop_keep_score": "",
                    "clip_hard_negative_score": "",
                    "clip_final_score": "",
                    "clip_global_final_score": "",
                    "clip_crop_final_score": "",
                    "clip_best_crop": "",
                    "clip_pred": "invalid_image",
                    "clip_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        valid_batch.append(image)
        valid_items.append({"row_idx": idx, "kind": "global", "crop_id": "global"})
        for crop_id, crop in make_crops(image, crop_cols, crop_rows, args.crop_overlap):
            valid_batch.append(crop)
            valid_items.append({"row_idx": idx, "kind": "crop", "crop_id": crop_id})

        if len(valid_batch) >= args.batch_size:
            flush_batch()

    flush_batch()

    for row_idx, bucket in row_scores.items():
        global_score = bucket.get("global")
        if not global_score:
            continue
        crop_scores = bucket.get("crops", [])
        best_crop = max(crop_scores, key=lambda item: item["final"]) if crop_scores else None
        best_crop_keep = best_crop["keep"] if best_crop else 0.0
        best_crop_final = best_crop["final"] if best_crop else 0.0

        positive_score = max(global_score["positive"], best_crop["positive"] if best_crop else 0.0)
        suspected_score = max(global_score["suspected"], best_crop["suspected"] if best_crop else 0.0)
        keep_score = max(global_score["keep"], best_crop_keep)
        negative_score = max(global_score["negative"], best_crop["negative"] if best_crop else 0.0)
        invalid_score = max(global_score["invalid"], best_crop["invalid"] if best_crop else 0.0)
        if best_crop and best_crop["final"] > global_score["final"]:
            hard_negative_score = best_crop["hard_negative"]
        else:
            hard_negative_score = global_score["hard_negative"]
        final_score = max(global_score["final"], best_crop_final)
        non_positive_score = max(negative_score, invalid_score, hard_negative_score)
        pred, margin = classify_row(
            keep_score,
            non_positive_score,
            invalid_score,
            final_score,
            args.margin_threshold,
            args.min_keep_score,
            args.final_threshold,
        )

        results[row_idx].update(
            {
                "clip_positive_score": f"{positive_score:.6f}",
                "clip_suspected_score": f"{suspected_score:.6f}",
                "clip_keep_score": f"{keep_score:.6f}",
                "clip_negative_score": f"{negative_score:.6f}",
                "clip_invalid_score": f"{invalid_score:.6f}",
                "clip_margin": f"{margin:.6f}",
                "clip_hard_negative_score": f"{hard_negative_score:.6f}",
                "clip_final_score": f"{final_score:.6f}",
                "clip_global_keep_score": f"{global_score['keep']:.6f}",
                "clip_crop_keep_score": f"{best_crop_keep:.6f}",
                "clip_global_final_score": f"{global_score['final']:.6f}",
                "clip_crop_final_score": f"{best_crop_final:.6f}",
                "clip_best_crop": best_crop["crop_id"] if best_crop else "",
                "clip_pred": pred,
                "clip_error": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(df.columns) + [
        "clip_positive_score",
        "clip_suspected_score",
        "clip_keep_score",
        "clip_negative_score",
        "clip_invalid_score",
        "clip_margin",
        "clip_hard_negative_score",
        "clip_final_score",
        "clip_global_keep_score",
        "clip_crop_keep_score",
        "clip_global_final_score",
        "clip_crop_final_score",
        "clip_best_crop",
        "clip_pred",
        "clip_error",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    result_df = pd.DataFrame(results)
    write_stats(stats_output_path, result_df, args, prompts)
    print(f"Done: {len(result_df)} rows -> {output_path}")
    print(f"Stats -> {stats_output_path}")
    print(result_df["clip_pred"].value_counts(dropna=False).to_string())

    if "human_label" in result_df.columns:
        risky = result_df[
            result_df["human_label"].isin(["positive", "suspected"])
            & result_df["clip_pred"].astype(str).str.startswith("drop")
        ]
        print(f"Risky drops among positive/suspected: {len(risky)}")
        if len(risky) > 0:
            print(risky[["sample_id", "human_label", "clip_pred", "clip_margin"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
