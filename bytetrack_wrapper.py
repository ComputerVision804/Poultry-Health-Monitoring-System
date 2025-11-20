# bytetrack_wrapper.py
# Try to adapt to whichever ByteTrack implementation you installed.
import numpy as np

class ByteTrackWrapper:
    def __init__(self, model_cfg=None, model_weights=None, frame_rate=30, device="cpu"):
        """
        Example usage:
            wrapper = ByteTrackWrapper()
            results = wrapper.update(detections, frame)
        Where detections = [(x1,y1,x2,y2,score), ...]
        """
        self.tracker = None
        # Try to import known implementations
        # Use importlib to avoid static import resolution errors and try multiple module paths
        import importlib
        YXByte = None
        try:
            for modname, attr in (
                ("yolox.tracker.byte_tracker", "BYTETracker"),
                ("yolox.tracker.byte_tracker.byte_tracker", "BYTETracker"),
                ("yolox.tracker", "BYTETracker"),
            ):
                try:
                    m = importlib.import_module(modname)
                    if hasattr(m, attr):
                        YXByte = getattr(m, attr)
                        break
                except Exception:
                    continue
            if YXByte is not None:
                self.tracker = YXByte(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
                self.impl = "yolox"
            else:
                raise ImportError("yolox BYTETracker not found")
        except Exception:
            try:
                # bytetrack-pytorch variant
                BT = None
                try:
                    m = importlib.import_module("bytetrack")
                    if hasattr(m, "BYTETracker"):
                        BT = getattr(m, "BYTETracker")
                except Exception:
                    BT = None
                if BT is not None:
                    self.tracker = BT()
                    self.impl = "bytetrack_pytorch"
                else:
                    raise ImportError("bytetrack BYTETracker not found")
            except Exception:
                self.tracker = None
                self.impl = None

        # internal mapping of active ids -> list
    def update(self, detections, ori_image=None):
        """
        detections: [(x1,y1,x2,y2,score), ...] in pixel coords
        returns: [{'id': id, 'bbox': (x1,y1,x2,y2), 'score': score}, ...]
        """
        if self.tracker is None:
            # fallback: naive assignment (copy input to new ids)
            out = []
            for i, d in enumerate(detections):
                out.append({'id': i+1, 'bbox': d[:4], 'score': d[4]})
            return out

        if self.impl == "yolox":
            # yolox BYTETracker expects tlwh, scores, and image info
            tlwhs = []
            scores = []
            for (x1,y1,x2,y2,s) in detections:
                w = x2 - x1
                h = y2 - y1
                tlwhs.append([x1, y1, w, h])
                scores.append(float(s))
            tlwhs = np.array(tlwhs, dtype=float)
            online_targets = self.tracker.update(tlwhs, scores, ori_image.shape)  # check exact signature
            outputs = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                x1,y1,w,h = tlwh
                x2 = x1 + w
                y2 = y1 + h
                outputs.append({'id': int(tid), 'bbox': (int(x1),int(y1),int(x2),int(y2)), 'score': float(t.score)})
            return outputs

        if self.impl == "bytetrack_pytorch":
            # adapt to that API accordingly
            online = self.tracker.update(detections)  # placeholder
            outputs = []
            for t in online:
                outputs.append({'id': int(t[0]), 'bbox': (int(t[1]),int(t[2]),int(t[3]),int(t[4])), 'score': float(t[5])})
            return outputs
