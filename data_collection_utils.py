import json
import os
from datetime import datetime
from typing import Iterable

import cv2
import numpy as np


def safe_word_folder(word: str) -> str:
    word = (word or "").strip()
    if not word:
        return ""
    word = word.replace(" ", "_")
    word = "".join(ch for ch in word if ch.isalnum() or ch in {"_", "-"})
    return word


def allowed_file(filename: str, allowed_ext: set[str]) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_ext


def make_sample_dir(base_dataset_dir: str, word: str, run_id: str) -> str:
    word_dir = os.path.join(base_dataset_dir, word)
    sample_dir = os.path.join(word_dir, run_id)
    os.makedirs(os.path.join(sample_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, "processed"), exist_ok=True)
    return sample_dir


def expand_hand_bbox(
    hands: Iterable[dict],
    frame_shape: tuple[int, int, int],
    offset: int,
    extra_ratio: float = 0.35,
) -> tuple[int, int, int, int]:
    x_min = min(h["bbox"][0] for h in hands)
    y_min = min(h["bbox"][1] for h in hands)
    x_max = max(h["bbox"][0] + h["bbox"][2] for h in hands)
    y_max = max(h["bbox"][1] + h["bbox"][3] for h in hands)

    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    extra = int(max(width, height) * extra_ratio)

    x1 = max(0, x_min - offset - extra)
    y1 = max(0, y_min - offset - extra)
    x2 = min(frame_shape[1], x_max + offset + extra)
    y2 = min(frame_shape[0], y_max + offset + extra)

    return x1, y1, x2, y2


def process_hand_to_white(
    frame_bgr: np.ndarray,
    detector,
    offset: int,
    img_size: int,
    extra_ratio: float = 0.35,
    hands: list[dict] | None = None,
    use_largest_only: bool = True,
) -> tuple[np.ndarray | None, bool]:
    if hands is None:
        hands, _ = detector.findHands(frame_bgr, draw=False)
    if not hands:
        return None, False

    if use_largest_only:
        largest = max(hands, key=lambda h: h["bbox"][2] * h["bbox"][3])
        chosen_hands = [largest]
    else:
        chosen_hands = hands

    x1, y1, x2, y2 = expand_hand_bbox(chosen_hands, frame_bgr.shape, offset, extra_ratio)

    img_crop = frame_bgr[y1:y2, x1:x2]
    if img_crop.size == 0:
        return None, False

    img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = h / w if w != 0 else 1.0

    if aspect_ratio > 1:
        k = img_size / h if h != 0 else 1.0
        w_cal = int(round(k * w))
        if w_cal <= 0:
            return None, False
        img_resize = cv2.resize(img_crop, (w_cal, img_size))
        w_gap = (img_size - w_cal) // 2
        img_white[:, w_gap:w_gap + w_cal] = img_resize
    else:
        k = img_size / w if w != 0 else 1.0
        h_cal = int(round(k * h))
        if h_cal <= 0:
            return None, False
        img_resize = cv2.resize(img_crop, (img_size, h_cal))
        h_gap = (img_size - h_cal) // 2
        img_white[h_gap:h_gap + h_cal, :] = img_resize

    return img_white, True


def extract_frames_from_video(
    video_path: str,
    sample_dir: str,
    detector,
    offset: int,
    img_size: int,
    sample_fps: int = 10,
    target_frames: int = 0,
    save_raw: bool = True,
    save_processed: bool = True,
    extra_ratio: float = 0.35,
    target_img_size: int = 224,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0

    interval = max(1, int(round(video_fps / max(1, sample_fps))))

    raw_dir = os.path.join(sample_dir, "raw")
    proc_dir = os.path.join(sample_dir, "processed")

    total_frames = 0
    saved_raw = 0
    saved_proc = 0
    skipped = 0
    target_frames = max(0, int(target_frames))

    frame_idx = 0
    last_raw_frame = None
    last_proc_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1

        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        hands, _ = detector.findHands(frame, draw=False)

        frame_name = f"frame_{saved_raw + 1:06d}.jpg"

        img_white, detected = process_hand_to_white(
            frame, detector, offset, target_img_size, extra_ratio, hands=hands, use_largest_only=True
        )

        if save_raw and detected and img_white is not None:
            cv2.imwrite(os.path.join(raw_dir, frame_name), img_white)
            saved_raw += 1
            last_raw_frame = img_white

        if save_processed:
            if detected and img_white is not None:
                cv2.imwrite(os.path.join(proc_dir, frame_name), img_white)
                saved_proc += 1
                last_proc_frame = img_white
            else:
                skipped += 1

        frame_idx += 1

        if target_frames:
            raw_done = (not save_raw) or (saved_raw >= target_frames)
            proc_done = (not save_processed) or (saved_proc >= target_frames)
            if raw_done and proc_done:
                break

    # Top up to the requested target using the last available frame to allow repeats.
    if target_frames > 0:
        if save_raw and last_raw_frame is not None:
            while saved_raw < target_frames:
                saved_raw += 1
                frame_name = f"frame_{saved_raw:06d}.jpg"
                cv2.imwrite(os.path.join(raw_dir, frame_name), last_raw_frame)
        if save_processed:
            source_proc_frame = last_proc_frame if last_proc_frame is not None else last_raw_frame
            if source_proc_frame is not None:
                if source_proc_frame.shape[0] != target_img_size or source_proc_frame.shape[1] != target_img_size:
                    source_proc_frame = cv2.resize(source_proc_frame, (target_img_size, target_img_size))
                while saved_proc < target_frames:
                    saved_proc += 1
                    frame_name = f"frame_{saved_proc:06d}.jpg"
                    cv2.imwrite(os.path.join(proc_dir, frame_name), source_proc_frame)

    cap.release()

    meta = {
        "video_path": os.path.basename(video_path),
        "video_fps": float(video_fps),
        "sample_fps": int(sample_fps),
        "interval": int(interval),
        "total_frames_read": int(total_frames),
        "raw_frames_saved": int(saved_raw),
        "processed_frames_saved": int(saved_proc),
        "processed_frames_skipped_no_hand": int(skipped),
        "imgSize": int(target_img_size),
        "offset": int(offset),
        "target_frames_requested": int(target_frames),
        "target_img_size": int(target_img_size),
        "created_at": datetime.now().isoformat(),
    }

    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
