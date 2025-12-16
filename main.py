import argparse
import base64
import json
import os
from datetime import datetime

import cv2
import numpy as np
import requests
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


OUTPUT_FOLDER = "tracker_clips"

# 單人裁切影片的固定尺寸（寬, 高）
CROP_SIZE = (400, 400)
# bbox 外的 padding，避免裁太緊
CROP_PADDING = 20


GEMMA_PROMPT = """
You will receive an image of a single person in a retail / checkout environment.
Please describe this customer in English in a SHORT, structured way, focusing on:
- gender (male / female / unknown)
- estimated age group (teen, 20s, 30s, 40s, 50+, or unknown)
- clothing and appearance (colors and main items, e.g. \"black jacket, blue jeans\")
- role / identity guess (customer, staff, delivery, other, or unknown)

Format your response as ONE short sentence, no line breaks, no bullet points.
If you really cannot see a person, respond with exactly: \"unknown\".
"""


def describe_person_with_gemma(image: np.ndarray) -> str | None:
    """
    Call local Gemma3 (via Ollama) to get a short description of a person.
    """
    try:
        # Encode image as JPEG and base64
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            return None
        jpeg_bytes = encoded.tobytes()
        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

        payload = {
            "model": "gemma3",
            "messages": [
                {
                    "role": "user",
                    "content": GEMMA_PROMPT,
                    "images": [b64_image],
                }
            ],
            "stream": False,
        }

        response = requests.post(
            "http://localhost:11434/api/chat", json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        text = (data.get("message", {}).get("content") or "").strip()
        if not text or text.lower() == "unknown":
            return None
        return text
    except Exception:
        # 靜默失敗，避免影響主流程
        return None


def crop_person_frame(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """
    依照給定 bbox 從整張畫面裁切出單人畫面，並縮放到固定大小 CROP_SIZE。

    這裡會：
    - 對 bbox 加上一點 padding
    - 做邊界檢查
    - 直接 resize 到固定大小，方便寫入 VideoWriter
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1_p = max(0, x1 - CROP_PADDING)
    y1_p = max(0, y1 - CROP_PADDING)
    x2_p = min(w, x2 + CROP_PADDING)
    y2_p = min(h, y2 + CROP_PADDING)

    if x2_p <= x1_p or y2_p <= y1_p:
        return None

    crop = frame[y1_p:y2_p, x1_p:x2_p]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, CROP_SIZE)
    return crop_resized


def main(
    source_video_path: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: list[int],
) -> None:
    # 建立輸出資料夾，用於存放每個 tracker 的影片
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Video clips will be saved to: {os.path.abspath(OUTPUT_FOLDER)}")

    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    # 追蹤每個人（tracker_id）在各個 zone 的資訊與 Gemma 描述與影片寫入器
    # 結構:
    # {
    #   tracker_id: {
    #       "zones": {
    #           zone_idx: {
    #               "entered_at": datetime,
    #               "last_seen_at": datetime,
    #               "exited_at": datetime | None,
    #               "dwell_time": float,
    #               "best_area": float,
    #               "best_frame": np.ndarray,
    #               "description": str | None,
    #               "logged": bool,
    #               "writer": cv2.VideoWriter | None,
    #           }
    #       }
    #   }
    # }
    person_memory: dict[int, dict] = {}

    # 每個 zone 當前在裡面的 tracker_id 集合，用來偵測進出
    active_in_zone: list[set[int]] = [set() for _ in zones]

    try:
        for frame in frames_generator:
            results = model(frame, verbose=False, device=device, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[find_in_list(detections.class_id, classes)]
            detections = detections.with_nms(threshold=iou)
            detections = tracker.update_with_detections(detections)

            annotated_frame = frame.copy()
            frame_time = datetime.now()

            for idx, zone in enumerate(zones):
                annotated_frame = sv.draw_polygon(
                    scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                )

                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                # 更新 person_memory：進入 / 停留 / 最佳畫面，並為每個 tracker 準備影片寫入器
                current_ids: set[int] = set()
                for det_idx, (tracker_id, dwell_time) in enumerate(
                    zip(detections_in_zone.tracker_id, time_in_zone)
                ):
                    if tracker_id is None:
                        continue
                    t_id = int(tracker_id)
                    current_ids.add(t_id)

                    # 初始化此人
                    mem = person_memory.setdefault(t_id, {"zones": {}})
                    zone_mem = mem["zones"].setdefault(
                        idx,
                        {
                            "entered_at": frame_time,
                            "last_seen_at": frame_time,
                            "exited_at": None,
                            "dwell_time": 0.0,
                            "best_area": 0.0,
                            "best_frame": None,
                            "description": None,
                            "logged": False,
                            "writer": None,
                        },
                    )

                    zone_mem["last_seen_at"] = frame_time
                    zone_mem["dwell_time"] = float(dwell_time)

                    # 更新最佳畫面（加入 padding，讓 Gemma 有更多上下文）
                    x1, y1, x2, y2 = map(int, detections_in_zone.xyxy[det_idx])
                    h, w, _ = frame.shape
                    x1_p = max(0, x1 - CROP_PADDING)
                    y1_p = max(0, y1 - CROP_PADDING)
                    x2_p = min(w, x2 + CROP_PADDING)
                    y2_p = min(h, y2 + CROP_PADDING)

                    area = float(max(0, x2 - x1) * max(0, y2 - y1))
                    if area > zone_mem.get("best_area", 0.0) and (x2 - x1) > 0 and (y2 - y1) > 0:
                        zone_mem["best_area"] = area
                        zone_mem["best_frame"] = frame[y1_p:y2_p, x1_p:x2_p].copy()

                    # 準備影片寫入器（每個 tracker_id + zone 一個檔案，輸出裁切後畫面）
                    if zone_mem["writer"] is None:
                        file_name = f"tracker_{t_id}_zone_{idx}.avi"
                        file_path = os.path.join(OUTPUT_FOLDER, file_name)
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        zone_mem["writer"] = cv2.VideoWriter(
                            file_path,
                            fourcc,
                            video_info.fps,
                            CROP_SIZE,
                        )

                    # 若停留時間 >= 5 秒，且尚未取得描述，呼叫 Gemma
                    if (
                        zone_mem["dwell_time"] >= 5.0
                        and zone_mem.get("description") is None
                        and zone_mem.get("best_frame") is not None
                    ):
                        description = describe_person_with_gemma(zone_mem["best_frame"])
                        zone_mem["description"] = description

                # 偵測離開 zone 的人，並在離開時輸出 JSON 記錄與關閉影片寫入器
                prev_ids = active_in_zone[idx]
                exited_ids = prev_ids - current_ids
                for t_id in exited_ids:
                    mem = person_memory.get(t_id)
                    if not mem:
                        continue
                    zone_mem = mem["zones"].get(idx)
                    if not zone_mem or zone_mem.get("logged"):
                        continue

                    # 關閉影片寫入器
                    if zone_mem.get("writer") is not None:
                        zone_mem["writer"].release()
                        zone_mem["writer"] = None

                    zone_mem["exited_at"] = frame_time
                    dwell = float(zone_mem.get("dwell_time", 0.0))

                    # 只記錄停留 >= 5 秒的人，避免路過
                    if dwell < 5.0:
                        zone_mem["logged"] = True
                        continue

                    # 若尚未有描述，離開時最後再嘗試一次
                    if (
                        zone_mem.get("description") is None
                        and zone_mem.get("best_frame") is not None
                    ):
                        description = describe_person_with_gemma(zone_mem["best_frame"])
                        zone_mem["description"] = description

                    video_file = f"tracker_{t_id}_zone_{idx}.avi"
                    record = {
                        "global_timestamp": datetime.now().isoformat(),
                        "tracker_id": t_id,
                        "zone_index": idx,
                        "entered_at": zone_mem["entered_at"].isoformat()
                        if isinstance(zone_mem.get("entered_at"), datetime)
                        else None,
                        "exited_at": zone_mem["exited_at"].isoformat()
                        if isinstance(zone_mem.get("exited_at"), datetime)
                        else None,
                        "dwell_time_seconds": dwell,
                        "person_description": zone_mem.get("description"),
                        "video_file": os.path.join(OUTPUT_FOLDER, video_file),
                    }

                    # 以 JSONL 方式不斷追加輸出
                    try:
                        with open("people_zone_log.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception:
                        # 寫檔失敗時忽略，避免中斷主流程
                        pass

                    zone_mem["logged"] = True

                active_in_zone[idx] = current_ids

                # 在畫面上繪製偵測框與文字標籤（即時顯示用）
                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )

                # 為每個 tracker_id 添加原本的停留時間標籤 + Gemma 描述（若已取得）
                labels: list[str] = []
                for tracker_id, dwell_time in zip(
                    detections_in_zone.tracker_id, time_in_zone
                ):
                    base_label = f"#{tracker_id} {int(dwell_time // 60):02d}:{int(dwell_time % 60):02d}"

                    desc = None
                    if tracker_id is not None:
                        t_id = int(tracker_id)
                        mem = person_memory.get(t_id)
                        if mem:
                            zone_mem = mem["zones"].get(idx)
                            if zone_mem:
                                desc = zone_mem.get("person_description") or zone_mem.get(
                                    "description"
                                )

                    if desc:
                        short_desc = desc.replace("\n", " ")
                        if len(short_desc) > 60:
                            short_desc = short_desc[:57] + "..."
                        base_label = f"{base_label} | {short_desc}"

                    labels.append(base_label)

                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )

                # 將「裁切後的個人畫面」寫入各自的影片
                for det_idx, tracker_id in enumerate(detections_in_zone.tracker_id):
                    if tracker_id is None:
                        continue
                    t_id = int(tracker_id)
                    mem = person_memory.get(t_id)
                    if not mem:
                        continue
                    zone_mem = mem["zones"].get(idx)
                    writer = zone_mem.get("writer") if zone_mem else None
                    if writer is None:
                        continue

                    box = detections_in_zone.xyxy[det_idx]
                    crop_frame = crop_person_frame(annotated_frame, box)
                    if crop_frame is not None:
                        writer.write(crop_frame)

            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 清理所有尚未釋放的影片寫入器
        for mem in person_memory.values():
            for zone_mem in mem["zones"].values():
                writer = zone_mem.get("writer")
                if writer is not None:
                    writer.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8s.pt",
        help="Path to the model weights file. Default is 'yolov8s.pt'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
