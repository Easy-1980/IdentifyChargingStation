import argparse
import csv
import math
import time
from pathlib import Path

import requests


GEOCONV_URL = "https://api.map.baidu.com/geoconv/v2/"
PANORAMA_URL = "http://api.uls.com.cn/panorama/v2"

# 坐标转换 AK：用于 https://api.map.baidu.com/geoconv/v2/
GEOCONV_AK = "c0q0Gb7bQXCXXXXOl4CCXXXXWEIPBAY"

# 全景下载 AK：用于 http://api.uls.com.cn/panorama/v2
PANORAMA_AK = "DXXXXAXZXX8W"

# 每个采样点下载 4 个方向的街景图，heading 是水平朝向角度。
HEADINGS = [0, 90, 180, 270]

# True: 调用 GEOCONV_URL 做 BD09LL <-> BD09MC 转换。
# False: 使用本地近似米制坐标采样。
USE_GEOCONV_API = True

EARTH_RADIUS_M = 6378137.0


def read_control_points(input_path):
    """读取控制点 CSV，并按 road_id 分组。"""
    roads = {}
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "station_id",
            "station_name",
            "road_id",
            "point_name",
            "point_order",
            "lng_bd09",
            "lat_bd09",
            "is_shared_point",
            "shared_with",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("输入 CSV 缺少字段: " + ",".join(sorted(missing)))

        for row in reader:
            road_id = row["road_id"].strip()
            point = {
                "station_id": row["station_id"].strip(),
                "station_name": row["station_name"].strip(),
                "road_id": road_id,
                "point_name": row["point_name"].strip(),
                "point_order": int(row["point_order"]),
                "lng": float(row["lng_bd09"]),
                "lat": float(row["lat_bd09"]),
                "is_shared_point": row["is_shared_point"].strip(),
                "shared_with": row["shared_with"].strip(),
            }
            roads.setdefault(road_id, []).append(point)

    for road_id in roads:
        roads[road_id].sort(key=lambda p: p["point_order"])
    return roads


def chunk_list(items, size):
    """百度坐标转换接口单次不宜提交太多点，这里按批拆分。"""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def baidu_convert(points, ak, model, sleep_seconds=0.05):
    """调用百度坐标转换 API。

    model=3: BD09LL 转 BD09MC
    model=4: BD09MC 转 BD09LL
    """
    converted = []
    for batch in chunk_list(points, 100):
        coords = ";".join(f"{lng},{lat}" for lng, lat in batch)
        params = {
            "ak": ak,
            "coords": coords,
            "model": model,
            "output": "json",
        }
        resp = requests.get(GEOCONV_URL, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"百度坐标转换 HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        if data.get("status") != 0:
            raise RuntimeError(f"百度坐标转换失败: {data}")

        for item in data.get("result", []):
            converted.append((float(item["x"]), float(item["y"])))

        time.sleep(sleep_seconds)

    if len(converted) != len(points):
        raise RuntimeError("百度坐标转换返回点数与请求点数不一致")
    return converted


def line_length(points_mc):
    """计算 BD09MC 折线总长度，单位米。"""
    total = 0.0
    for i in range(1, len(points_mc)):
        x1, y1 = points_mc[i - 1]
        x2, y2 = points_mc[i]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def interpolate_at_distance(points_mc, target_distance):
    """在折线上按累计距离插值。"""
    if target_distance <= 0:
        return points_mc[0]

    walked = 0.0
    for i in range(1, len(points_mc)):
        x1, y1 = points_mc[i - 1]
        x2, y2 = points_mc[i]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len == 0:
            continue

        if walked + seg_len >= target_distance:
            ratio = (target_distance - walked) / seg_len
            return (x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio)

        walked += seg_len

    return points_mc[-1]


def make_sample_distances(total_length, spacing):
    """生成 0、spacing、2*spacing ... 形式的采样距离。"""
    distances = []
    d = 0.0
    while d <= total_length + 1e-6:
        distances.append(round(d, 3))
        d += spacing
    end_distance = round(total_length, 3)
    if not distances:
        distances.append(0.0)
    if abs(distances[-1] - end_distance) > 1e-6:
        distances.append(end_distance)
    return distances


def lnglat_to_local_meters(points_ll):
    """把经纬度近似投影到本地米制坐标，避免依赖不可用的 geoconv 接口。"""
    origin_lng, origin_lat = points_ll[0]
    origin_lat_rad = math.radians(origin_lat)
    cos_lat = math.cos(origin_lat_rad)
    points_m = []

    for lng, lat in points_ll:
        x = math.radians(lng - origin_lng) * EARTH_RADIUS_M * cos_lat
        y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
        points_m.append((x, y))

    return points_m, (origin_lng, origin_lat, cos_lat)


def local_meters_to_lnglat(points_m, origin):
    """把本地米制坐标转回经纬度。"""
    origin_lng, origin_lat, cos_lat = origin
    points_ll = []

    for x, y in points_m:
        lng = origin_lng + math.degrees(x / (EARTH_RADIUS_M * cos_lat))
        lat = origin_lat + math.degrees(y / EARTH_RADIUS_M)
        points_ll.append((lng, lat))

    return points_ll


def build_samples(roads, geoconv_ak, spacing):
    """生成所有道路的采样点，并转回 BD09LL。"""
    all_samples = []

    for road_id, points in roads.items():
        if len(points) < 2:
            print(f"跳过 {road_id}: 控制点少于 2 个")
            continue

        station_id = points[0]["station_id"]
        station_name = points[0]["station_name"]
        start_lng = points[0]["lng"]
        start_lat = points[0]["lat"]
        end_lng = points[-1]["lng"]
        end_lat = points[-1]["lat"]
        bd09ll_points = [(p["lng"], p["lat"]) for p in points]
        if USE_GEOCONV_API:
            points_mc = baidu_convert(bd09ll_points, ak=geoconv_ak, model=3)
            local_origin = None
        else:
            points_mc, local_origin = lnglat_to_local_meters(bd09ll_points)

        total_length = line_length(points_mc)
        distances = make_sample_distances(total_length, spacing)
        sample_points_mc = [interpolate_at_distance(points_mc, d) for d in distances]
        if USE_GEOCONV_API:
            sample_points_ll = baidu_convert(sample_points_mc, ak=geoconv_ak, model=4)
        else:
            sample_points_ll = local_meters_to_lnglat(sample_points_mc, local_origin)

        for idx, (distance_m, (lng, lat)) in enumerate(zip(distances, sample_points_ll)):
            sample_id = f"{road_id}_{int(round(distance_m)):04d}"
            if idx == 0:
                dedup_lng = start_lng
                dedup_lat = start_lat
            elif idx == len(distances) - 1:
                dedup_lng = end_lng
                dedup_lat = end_lat
            else:
                dedup_lng = lng
                dedup_lat = lat

            all_samples.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "road_id": road_id,
                    "sample_id": sample_id,
                    "distance_m": f"{distance_m:.1f}",
                    "lng_bd09": f"{lng:.6f}",
                    "lat_bd09": f"{lat:.6f}",
                    "dedup_lng": f"{dedup_lng:.6f}",
                    "dedup_lat": f"{dedup_lat:.6f}",
                }
            )

    return all_samples


def download_panorama(sample, ak, image_dir, width, height, fov, pitch, heading):
    """下载单张百度全景静态图，失败时返回 failed。"""
    image_path = image_dir / f"{sample['sample_id']}.jpg"
    params = {
        "ak": ak,
        "width": width,
        "height": height,
        "location": f"{sample['lng_bd09']},{sample['lat_bd09']}",
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
    }

    try:
        resp = requests.get(PANORAMA_URL, params=params, timeout=20)
        if resp.status_code != 200:
            return image_path.as_posix(), "failed", f"HTTP {resp.status_code}: {resp.text[:200]}"

        content_type = resp.headers.get("Content-Type", "").lower()
        if "image" not in content_type:
            return image_path.as_posix(), "failed", resp.text[:200]

        image_path.write_bytes(resp.content)
        return image_path.as_posix(), "success", ""
    except requests.RequestException as exc:
        return image_path.as_posix(), "failed", str(exc)


def write_samples(output_path, samples):
    """写出采样结果 CSV。"""
    fieldnames = [
        "station_id",
        "station_name",
        "road_id",
        "sample_id",
        "distance_m",
        "lng_bd09",
        "lat_bd09",
        "heading",
        "image_path",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def parse_args():
    parser = argparse.ArgumentParser(description="百度道路控制点街景采样最小脚本")
    parser.add_argument("--input", default="road_control_points_new.csv", help="输入控制点 CSV")
    parser.add_argument("--output", default="sample_points_new.csv", help="输出采样点 CSV")
    parser.add_argument("--image-dir", default="images", help="图片保存目录")
    parser.add_argument("--geoconv-ak", default=GEOCONV_AK, help="坐标转换 AK，默认使用代码顶部的 GEOCONV_AK")
    parser.add_argument("--panorama-ak", default=PANORAMA_AK, help="全景下载 AK，默认使用代码顶部的 PANORAMA_AK")
    parser.add_argument("--spacing", type=float, default=20.0, help="采样间距，单位米")
    parser.add_argument("--width", type=int, default=1024, help="全景图宽度")
    parser.add_argument("--height", type=int, default=512, help="全景图高度")
    parser.add_argument("--fov", type=int, default=180, help="全景图视场角")
    parser.add_argument("--pitch", type=int, default=0, help="全景图俯仰角")
    parser.add_argument("--no-download", action="store_true", help="只生成采样点，不下载图片")
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        if args.spacing <= 0:
            raise ValueError("--spacing 必须大于 0")
        if USE_GEOCONV_API and (not args.geoconv_ak or args.geoconv_ak == "YOUR_GEOCONV_AK"):
            raise ValueError("请先在代码顶部把 GEOCONV_AK 替换成可用于坐标转换的 AK")
        if not args.panorama_ak or args.panorama_ak == "YOUR_PANORAMA_AK":
            raise ValueError("请先在代码顶部把 PANORAMA_AK 替换成可用于全景下载的 AK")

        input_path = Path(args.input)
        output_path = Path(args.output)
        image_dir = Path(args.image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        roads = read_control_points(input_path)
        samples = build_samples(roads, geoconv_ak=args.geoconv_ak, spacing=args.spacing)
        image_rows = []
        downloaded_cache = {}

        for sample in samples:
            for heading in HEADINGS:
                image_sample = sample.copy()
                image_sample["heading"] = heading
                image_sample["sample_id"] = f"{sample['sample_id']}_H{heading:03d}"

                image_path = image_dir / f"{image_sample['sample_id']}.jpg"
                image_sample["image_path"] = image_path.as_posix()
                dedup_key = (image_sample["dedup_lng"], image_sample["dedup_lat"], heading)

                if dedup_key in downloaded_cache:
                    image_sample["image_path"] = downloaded_cache[dedup_key]
                    print(f"复用图片 {image_sample['sample_id']} -> {image_sample['image_path']}")
                    image_rows.append(image_sample)
                    continue

                if args.no_download:
                    downloaded_cache[dedup_key] = image_sample["image_path"]
                else:
                    saved_path, status, error_msg = download_panorama(
                        image_sample,
                        ak=args.panorama_ak,
                        image_dir=image_dir,
                        width=args.width,
                        height=args.height,
                        fov=args.fov,
                        pitch=args.pitch,
                        heading=heading,
                    )
                    image_sample["image_path"] = saved_path
                    downloaded_cache[dedup_key] = saved_path
                    if status == "failed":
                        print(f"下载失败 {image_sample['sample_id']}: {error_msg}")
                    time.sleep(0.1)

                image_rows.append(image_sample)

        write_samples(output_path, image_rows)
        print(f"完成: 生成 {len(samples)} 个采样点，{len(image_rows)} 条图片记录，输出 {output_path}")
    except Exception as exc:
        raise SystemExit(f"运行失败: {exc}") from None


if __name__ == "__main__":
    main()
