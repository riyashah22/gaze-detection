import asyncio
import cv2
import csv
import numpy as np
import os
import logging
import yaml
from collections import defaultdict
from typing import Tuple
from openvino.runtime import Core
from typing import Optional


# --- Load config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE = cfg["device"]
H_THRESH = cfg["horizontal_threshold"]
V_THRESH_UP = cfg["vertical_threshold_up"]
V_THRESH_DOWN = cfg["vertical_threshold_down"]
MARGIN = cfg["margin"]
ARROW_SCALE = cfg["arrow_scale"]
NUM_CHUNKS = cfg["num_chunks"]
VIDEO_PATH = cfg["video_path"]
OUTPUT_CSV = cfg["output_csv"]

# --- Load models ---
def load_models(device: str) -> dict:
    ie = Core()
    model_paths = {
        "face": "intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml",
        "head_pose": "intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml",
        "landmarks": "intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
        "gaze": "intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml",
    }
    return {name: ie.compile_model(ie.read_model(path), device) for name, path in model_paths.items()}


# --- Preprocess helper ---
def preprocess(frame: np.ndarray, shape: tuple) -> np.ndarray:
    H, W = shape[2], shape[3]
    img = cv2.resize(frame, (W, H))
    return img.transpose(2, 0, 1)[None, ...].astype(np.float32)

def crop_eye(frame: np.ndarray, center: Tuple[int, int], size: int = 60) -> Optional[np.ndarray]:
    x, y = center
    hs = size // 2
    x1, x2 = max(0, x - hs), min(frame.shape[1], x + hs)
    y1, y2 = max(0, y - hs), min(frame.shape[0], y + hs)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (size, size)) if roi.shape[:2] != (size, size) else roi

# --- Gaze status inference per frame ---
def get_gaze_status(frame: np.ndarray, models: dict) -> str:
    out = frame.copy()
    H_img, W_img = frame.shape[:2]
    status = "NO_FACE"

    eff_H = H_THRESH + MARGIN
    eff_V_up = V_THRESH_UP + MARGIN
    eff_V_down = V_THRESH_DOWN + MARGIN

    face_blob = preprocess(frame, models["face"].input(0).shape)
    dets = models["face"]([face_blob])[models["face"].output(0)]

    for det in dets[0][0]:
        if float(det[2]) < 0.8:
            continue
        xmin, ymin, xmax, ymax = (det[3:] * [W_img, H_img, W_img, H_img]).astype(int)
        if xmax <= xmin or ymax <= ymin:
            continue

        face = frame[ymin:ymax, xmin:xmax]

        hp_blob = preprocess(face, models["head_pose"].input(0).shape)
        hp_outs = models["head_pose"]([hp_blob])
        yaw, pitch, roll = [hp_outs[k].flatten()[0] for k in sorted(hp_outs.keys())]

        lm_blob = preprocess(face, models["landmarks"].input(0).shape)
        lm = models["landmarks"]([lm_blob])[models["landmarks"].output(0)].reshape(-1)
        left_eye = (xmin + int(lm[0] * face.shape[1]), ymin + int(lm[1] * face.shape[0]))
        right_eye = (xmin + int(lm[2] * face.shape[1]), ymin + int(lm[3] * face.shape[0]))

        le_roi = crop_eye(frame, left_eye)
        re_roi = crop_eye(frame, right_eye)
        if le_roi is None or re_roi is None:
            return "NO_EYE_ROI"

        gz = models["gaze"]
        inputs = {
            "left_eye_image": preprocess(le_roi, gz.input(0).shape),
            "right_eye_image": preprocess(re_roi, gz.input(1).shape),
            "head_pose_angles": np.array([yaw, pitch, roll], np.float32).reshape(1, 3)
        }
        gv = next(iter(gz(inputs).values()))[0]
        dx, dy = -gv[0], -gv[1]

        horiz_ok = abs(dx) <= eff_H
        vert_ok = (dy >= 0 and dy <= eff_V_up) or (dy < 0 and abs(dy) <= eff_V_down)
        return "INSIDE" if (horiz_ok and vert_ok) else "OUTSIDE"

    return status

# --- Chunk Processing Function ---
def process_chunk(video_path: str, start: int, end: int, fps: float, chunk_id: int) -> str:
    models = load_models(DEVICE)
    print(f"[Chunk {chunk_id}] Processing frames {start}-{end}")
    per_sec = defaultdict(list)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for i in range(start, end):
        ret, frame = cap.read()
        if not ret: break

        status = get_gaze_status(frame, models)
        sec = int(i / fps)
        per_sec[sec].append(status)

    cap.release()

    path = f"temp_chunk_{chunk_id}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for sec in sorted(per_sec):
            counts = {s: per_sec[sec].count(s) for s in ("INSIDE", "OUTSIDE", "NO_FACE")}
            majority = max(counts, key=counts.get)
            m, s = divmod(sec, 60)
            writer.writerow([f"{m}.{s:02d}", majority])

    print(f"[Chunk {chunk_id}] CSV → {path}")
    return path

# --- Async wrapper ---
async def process_chunk_async(video_path: str, start: int, end: int, fps: float, chunk_id: int) -> str:
    return await asyncio.to_thread(process_chunk, video_path, start, end, fps, chunk_id)

# --- Main Async Entry ---
async def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunk_sz = total_frames // NUM_CHUNKS
    tasks = [
        process_chunk_async(
            VIDEO_PATH,
            i * chunk_sz,
            total_frames if i == NUM_CHUNKS - 1 else (i + 1) * chunk_sz,
            fps,
            i
        ) for i in range(NUM_CHUNKS)
    ]

    files = await asyncio.gather(*tasks)

    merged = {}
    for f in files:
        with open(f) as r:
            for ts, st in csv.reader(r): merged[ts] = st
        os.remove(f)

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "status"])
        for ts in sorted(merged): w.writerow([ts, merged[ts]])

    logging.info(f"Saved final CSV → {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
