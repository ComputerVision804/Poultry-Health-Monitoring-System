# logger_csv.py
import csv, threading, time
from typing import Dict, Any

class CSVLogger:
    def __init__(self, path="poultry_log.csv", header=True):
        self.path = path
        self.lock = threading.Lock()
        if header:
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","frame_idx","track_id","x1","y1","x2","y2","behavior","flow_mean","head_flow_mean"])

    def log_track(self, frame_idx:int, track_id:int, bbox:tuple, behavior:str, flow_mean:float, head_flow_mean:float):
        ts = time.time()
        row = [ts, frame_idx, track_id, *bbox, behavior, flow_mean, head_flow_mean]
        with self.lock:
            with open(self.path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(row)
