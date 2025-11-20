#!/usr/bin/env python3
"""
poultry_monitor_full.py

Full single-file prototype:
- YOLOv11 detection (Ultralytics)
- Optional SAM2 segmentation (Segment Anything) for masks
- Improved SimpleCentroidTracker (Hungarian or greedy matching)
- Dense optical flow micro-movement per tracked object
- Heuristic context-aware BehaviorClassifier
- 4-panel OpenCV live dashboard (annotated video, flow vis, perf panel, stats panel)

Usage:
    python poultry_monitor_full.py --source 0 --yolo yolo11s.pt --sam sam_vit_h.pth

Author: ChatGPT (GPT-5 Thinking mini)
Date: 2025-11-19
"""

import argparse
import time
import math
import numpy as np
import cv2
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import importlib

# -----------------------
# Optional imports & flags
# -----------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("ultralytics not found. Install: pip install ultralytics")

# SAM optional
SamPredictor = None
sam_model_registry = None
try:
    _sam_mod = importlib.import_module("segment_anything")
    SamPredictor = getattr(_sam_mod, "SamPredictor", None)
    sam_model_registry = getattr(_sam_mod, "sam_model_registry", None)
    if SamPredictor is None or sam_model_registry is None:
        SamPredictor = None
        sam_model_registry = None
        print("segment_anything imported but missing expected API (SamPredictor/sam_model_registry).")
except Exception:
    SamPredictor = None
    sam_model_registry = None
    print("segment_anything not installed. SAM segmentation disabled unless you install it.")

# Hungarian solver (scipy)
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_HUNGARIAN = True
except Exception:
    _HAS_HUNGARIAN = False

# -----------------------
# Config
# -----------------------
CONFIG = {
    "YOLO_WEIGHTS": "yolo11s.pt",
    "SAM_WEIGHTS": "sam_vit_h.pth",
    "DEVICE": "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
    "CONF_THRESHOLD": 0.35,
    "TRACKER_MAX_AGE": 30,
    "MAX_DISTANCE": 60.0,
    "OPTICAL_FLOW_PARAMS": dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
    "MAX_TRAIL": 30,
    "DISPLAY_RESIZE": (1280, 720),
    "INACTIVITY_SPEED_THRESHOLD": 1.5,
    "FEEDING_FLOW_THRESHOLD": 0.8,
    "HEAD_MOVEMENT_THRESHOLD": 0.8,
    "FPS_SMOOTHING_ALPHA": 0.15
}

# -----------------------
# Data classes
# -----------------------
@dataclass
class Track:
    id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    history: deque = field(default_factory=lambda: deque(maxlen=CONFIG["MAX_TRAIL"]))
    last_update_frame: int = 0
    last_speed: float = 0.0
    seg_mask: np.ndarray = None
    head_box: Tuple[int, int, int, int] = None
    behavior: str = "Unknown"
    meta: Dict[str, Any] = field(default_factory=dict)
    lost: int = 0

# -----------------------
# Improved SimpleCentroidTracker
# -----------------------
class SimpleCentroidTracker:
    def __init__(self, max_distance: float = CONFIG["MAX_DISTANCE"], max_lost: int = CONFIG["TRACKER_MAX_AGE"]):
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.max_distance = max_distance
        self.max_lost = max_lost

    @staticmethod
    def _centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[Tuple[int, int, int, int, float]]):
        """
        detections: list of (x1,y1,x2,y2,score)
        returns: list of {'id': id, 'bbox': (x1,y1,x2,y2), 'score':score}
        """
        results = []
        if len(detections) == 0:
            # increase lost for all tracks and delete stale
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
            return []

        det_boxes = [tuple(d[:4]) for d in detections]
        det_centroids = [self._centroid(b) for b in det_boxes]
        det_scores = [float(d[4]) for d in detections]

        if len(self.tracks) == 0:
            for i, box in enumerate(det_boxes):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "centroid": det_centroids[i], "lost": 0, "score": det_scores[i]}
                results.append({"id": tid, "bbox": box, "score": det_scores[i]})
            return results

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]

        cost_matrix = np.zeros((len(track_centroids), len(det_centroids)), dtype=np.float32)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                cost_matrix[i, j] = self._euclid(tc, dc)

        assigned_tracks = set()
        assigned_dets = set()

        if _HAS_HUNGARIAN:
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] <= self.max_distance:
                    tid = track_ids[r]
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)
                    box = det_boxes[c]
                    score = det_scores[c]
                    self.tracks[tid]["bbox"] = box
                    self.tracks[tid]["centroid"] = det_centroids[c]
                    self.tracks[tid]["lost"] = 0
                    self.tracks[tid]["score"] = score
                    results.append({"id": tid, "bbox": box, "score": score})
        else:
            pairs = []
            for i in range(cost_matrix.shape[0]):
                for j in range(cost_matrix.shape[1]):
                    pairs.append((cost_matrix[i, j], i, j))
            pairs.sort(key=lambda x: x[0])
            used_rows = set()
            used_cols = set()
            for dist, r, c in pairs:
                if r in used_rows or c in used_cols:
                    continue
                if dist <= self.max_distance:
                    tid = track_ids[r]
                    used_rows.add(r)
                    used_cols.add(c)
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)
                    box = det_boxes[c]
                    score = det_scores[c]
                    self.tracks[tid]["bbox"] = box
                    self.tracks[tid]["centroid"] = det_centroids[c]
                    self.tracks[tid]["lost"] = 0
                    self.tracks[tid]["score"] = score
                    results.append({"id": tid, "bbox": box, "score": score})

        # new detections -> new tracks
        for di, box in enumerate(det_boxes):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "centroid": det_centroids[di], "lost": 0, "score": det_scores[di]}
                results.append({"id": tid, "bbox": box, "score": det_scores[di]})

        # increment lost counter for unassigned tracks and remove stale
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]

        return results

# -----------------------
# YOLO wrapper
# -----------------------
class YOLODetector:
    def __init__(self, weight_path: str, device: str = "cpu", conf_thres: float = 0.4):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO required. pip install ultralytics")
        self.model = YOLO(weight_path)
        self.device = device
        self.conf_thres = conf_thres

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        frame: BGR image (OpenCV). Returns detections list of dicts with bbox (x1,y1,x2,y2), score, class
        """
        # ultralytics typically expects RGB
        img = frame[..., ::-1]
        res = self.model.predict(img, device=self.device, imgsz=640, conf=self.conf_thres, verbose=False)
        results = []
        if len(res) == 0:
            return results
        r = res[0]
        # robust attribute access
        boxes = getattr(r.boxes, "xyxy", None)
        confs = getattr(r.boxes, "conf", None)
        cls = getattr(r.boxes, "cls", None)
        if boxes is None:
            return results
        boxes = boxes.cpu().numpy()
        confs = confs.cpu().numpy() if confs is not None else np.ones(len(boxes))
        cls = cls.cpu().numpy() if cls is not None else np.zeros(len(boxes))
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = float(confs[i])
            c = int(cls[i])
            results.append({"bbox": (x1, y1, x2, y2), "score": score, "class": c})
        return results

# -----------------------
# SAM wrapper (optional)
# -----------------------
class SAMSegmenter:
    def __init__(self, sam_weights: str):
        if SamPredictor is None or sam_model_registry is None:
            raise RuntimeError("Segment Anything (SAM) required; install segment-anything.")
        # registry expects key names like "vit_h"
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_weights)
        self.predictor = SamPredictor(self.sam)

    def segment(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        self.predictor.set_image(frame)
        masks = []
        for (x1, y1, x2, y2) in boxes:
            box = np.array([x1, y1, x2, y2])
            masks_result, scores, logits = self.predictor.predict(box=box[None, :], multimask_output=False)
            mask = masks_result[0].astype(np.uint8)
            masks.append(mask)
        return masks

# -----------------------
# Optical Flow Analyzer
# -----------------------
class OpticalFlowAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def compute_dense(self, gray: np.ndarray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            # return zero flow and zero mean
            empty_vis = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            return empty_vis, np.zeros_like(gray, dtype=np.float32), 0.0

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # visualization
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.prev_gray = gray.copy()
        mean_mag = float(np.mean(mag))
        return vis, mag, mean_mag

# -----------------------
# Behavior classifier (heuristic)
# -----------------------
class BehaviorClassifier:
    def classify(self, track: Track, body_flow_mean: float, head_flow_mean: float) -> str:
        speed = track.last_speed
        if speed > 6.0:
            return "Running"
        if 1.5 < speed <= 6.0:
            return "Walking"
        if speed <= CONFIG["INACTIVITY_SPEED_THRESHOLD"]:
            if body_flow_mean > CONFIG["FEEDING_FLOW_THRESHOLD"] and head_flow_mean > CONFIG["HEAD_MOVEMENT_THRESHOLD"]:
                return "Feeding"
            if head_flow_mean > CONFIG["HEAD_MOVEMENT_THRESHOLD"]:
                return "Alert"
            if body_flow_mean > 0.2:
                return "Resting"
            return "Inactive"
        return "Inactive"

# -----------------------
# Drawing / Dashboard helpers
# -----------------------
def draw_tracks_on_frame(frame: np.ndarray, tracks: Dict[int, Track]) -> np.ndarray:
    out = frame.copy()
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = map(int, tr.bbox)
        color = (int((tid * 37) % 255), int((tid * 79) % 255), int((tid * 149) % 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID:{tid} {tr.behavior}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if tr.history:
            pts = list(tr.history)
            for i in range(1, len(pts)):
                cv2.line(out, (int(pts[i-1][0]), int(pts[i-1][1])), (int(pts[i][0]), int(pts[i][1])), color, 2)
            cx, cy = int(tr.centroid[0]), int(tr.centroid[1])
            cv2.circle(out, (cx, cy), 4, color, -1)
    return out

def draw_perf_panel(fps, processing_time_ms, counters: Dict[str, int], panel_size=(640, 360)):
    img = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
    cv2.putText(img, f"Proc(ms): {processing_time_ms:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
    y = 120
    for k, v in counters.items():
        cv2.putText(img, f"{k}: {v}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        y += 30
    # simple placeholder trend
    pts = np.array([[20 + i*8, int(panel_size[1] - 20 - (i%10)*6)] for i in range(60)], dtype=np.int32)
    cv2.polylines(img, [pts], False, (50, 200, 50), 2)
    return img

def draw_stats_panel(tracks: Dict[int, Track], panel_size=(640,360)):
    img = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    dist = defaultdict(int)
    for tr in tracks.values():
        dist[tr.behavior] += 1
    y = 30
    cv2.putText(img, "Behavior distribution:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
    y += 30
    for k, v in dist.items():
        cv2.putText(img, f"{k}: {v}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
        y += 30
    total = sum(dist.values()) if sum(dist.values()) > 0 else 1
    healthy = dist.get("Feeding",0) + dist.get("Walking",0) + dist.get("Running",0)
    score = int(100 * (healthy / total))
    cv2.putText(img, f"Flock Health Score: {score}%", (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,220,100), 2)
    return img

def create_four_panel(frame, perf_panel, flow_vis, stats_panel, sizes=CONFIG["DISPLAY_RESIZE"]):
    w, h = sizes
    half_w, half_h = w//2, h//2
    def rs(img, tw=half_w, th=half_h):
        try:
            return cv2.resize(img, (tw, th))
        except Exception:
            return np.zeros((th, tw, 3), dtype=np.uint8)
    a = rs(frame)
    b = rs(flow_vis) if flow_vis is not None else np.zeros_like(a)
    c = rs(perf_panel)
    d = rs(stats_panel)
    top = np.hstack([a, b])
    bottom = np.hstack([c, d])
    board = np.vstack([top, bottom])
    return board

# -----------------------
# Main loop
# -----------------------
def main_loop(source=0, yolo_weights=None, sam_weights=None, show=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Models
    detector = None
    segmenter = None
    if yolo_weights:
        detector = YOLODetector(yolo_weights, device=CONFIG["DEVICE"], conf_thres=CONFIG["CONF_THRESHOLD"])
    else:
        print("No YOLO weights provided. Provide --yolo <weights>.")

    if sam_weights and SamPredictor is not None:
        try:
            segmenter = SAMSegmenter(sam_weights)
        except Exception as e:
            print("SAM init failed:", e)
            segmenter = None
    else:
        segmenter = None

    # tracker & helpers
    tracker = SimpleCentroidTracker(max_distance=CONFIG["MAX_DISTANCE"], max_lost=CONFIG["TRACKER_MAX_AGE"])
    flow_analyzer = OpticalFlowAnalyzer()
    behavior_clf = BehaviorClassifier()

    tracks: Dict[int, Track] = {}
    frame_idx = 0
    fps_smoothed = 0.0
    last_time = time.time()
    counters = defaultdict(int)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot fetch frame.")
            break

        # Resize for processing/display
        display_w, display_h = CONFIG["DISPLAY_RESIZE"]
        frame_disp = cv2.resize(frame, (display_w, display_h))
        gray = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)

        # Detection
        detections = []
        if detector:
            dets = detector.detect(frame_disp)
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                score = d["score"]
                detections.append((x1, y1, x2, y2, score))

        # Tracking
        tr_results = tracker.update(detections)
        current_ids = set()
        for res in tr_results:
            tid = res["id"]
            bbox = res["bbox"]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_ids.add(tid)
            if tid not in tracks:
                tr = Track(id=tid, bbox=bbox, centroid=(cx, cy))
                tr.history.append((cx, cy))
                tr.last_update_frame = frame_idx
                tracks[tid] = tr
            else:
                tr = tracks[tid]
                prev_centroid = tr.centroid
                dx = cx - prev_centroid[0]
                dy = cy - prev_centroid[1]
                speed = math.hypot(dx, dy)
                tr.last_speed = speed
                tr.bbox = bbox
                tr.centroid = (cx, cy)
                tr.history.append((cx, cy))
                tr.last_update_frame = frame_idx
                tr.lost = 0

        # Remove stale tracks not seen for a while
        stale_ids = [tid for tid in list(tracks.keys()) if tid not in current_ids and frame_idx - tracks[tid].last_update_frame > CONFIG["TRACKER_MAX_AGE"]]
        for sid in stale_ids:
            del tracks[sid]

        # Segmentation (if available)
        boxes_for_sam = [tracks[tid].bbox for tid in tracks.keys()]
        track_id_order = list(tracks.keys())
        masks = []
        if segmenter and len(boxes_for_sam) > 0:
            try:
                masks = segmenter.segment(frame_disp, boxes_for_sam)
            except Exception as e:
                print("SAM segmentation error:", e)
                masks = [None] * len(boxes_for_sam)
        else:
            masks = [None] * len(boxes_for_sam)

        # Optical flow
        flow_vis, mag_map, global_mean = flow_analyzer.compute_dense(gray)

        # Per-track flow/heads & classification
        for idx, tid in enumerate(track_id_order):
            tr = tracks[tid]
            mask = masks[idx] if idx < len(masks) else None
            if mask is not None and mag_map is not None:
                mask_bool = mask.astype(bool)
                if mask_bool.sum() > 0:
                    body_mean = float(np.mean(mag_map[mask_bool]))
                else:
                    body_mean = 0.0
            else:
                body_mean = 0.0

            # approximate head region (top 25% of bbox)
            x1, y1, x2, y2 = map(int, tr.bbox)
            h = max(1, y2 - y1)
            head_y1 = y1
            head_y2 = y1 + max(1, int(0.25 * h))
            head_x1 = x1
            head_x2 = x2
            head_y2 = min(head_y2, display_h - 1)
            head_mask = np.zeros((display_h, display_w), dtype=np.uint8)
            head_mask[head_y1:head_y2, head_x1:head_x2] = 1
            if mag_map is not None and head_mask.sum() > 0:
                head_mean = float(np.mean(mag_map[head_mask.astype(bool)]))
            else:
                head_mean = 0.0

            tr.meta["flow_mean"] = body_mean
            tr.meta["head_flow_mean"] = head_mean
            tr.behavior = behavior_clf.classify(tr, body_mean, head_mean)

        # Build panels
        annotated = draw_tracks_on_frame(frame_disp, tracks)
        perf_panel = draw_perf_panel(fps_smoothed, (time.time() - start_time) * 1000.0, counters)
        stats_panel = draw_stats_panel(tracks)
        composed = create_four_panel(annotated, perf_panel, flow_vis if flow_vis is not None else np.zeros_like(annotated), stats_panel, sizes=CONFIG["DISPLAY_RESIZE"])

        if show:
            cv2.imshow("Poultry Health Monitor - 4 Panel", composed)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        # FPS smoothing
        now = time.time()
        inst_fps = 1.0 / (now - last_time) if now != last_time else 0.0
        fps_smoothed = (1 - CONFIG["FPS_SMOOTHING_ALPHA"]) * fps_smoothed + CONFIG["FPS_SMOOTHING_ALPHA"] * inst_fps if fps_smoothed > 0 else inst_fps
        last_time = now
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poultry Health Monitoring - Full Prototype")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for camera or path)")
    parser.add_argument("--yolo", type=str, default=CONFIG["YOLO_WEIGHTS"], help="YOLO weights (e.g. yolo11s.pt)")
    parser.add_argument("--sam", type=str, default=None, help="SAM weights (optional)")
    args = parser.parse_args()

    src = int(args.source) if (args.source.isdigit()) else args.source
    try:
        main_loop(source=src, yolo_weights=args.yolo, sam_weights=args.sam, show=True)
    except Exception as e:
        print("Fatal error:", e)
        raise
