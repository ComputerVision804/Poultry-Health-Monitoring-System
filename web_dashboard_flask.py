# web_dashboard_flask.py
# Requirements: flask, opencv-python, requests
# Run: python web_dashboard_flask.py
from flask import Flask, Response, jsonify, request
import threading, time, cv2, io
from typing import Optional
from alerts import send_email_alert, send_sms_alert  # from section 4 below
from logger_csv import CSVLogger  # from section 3 below

app = Flask(__name__)

# Thread-safe global frame holder
FRAME_LOCK = threading.Lock()
LATEST_FRAME = None  # bytes of JPEG
LATEST_STATS = {}

# Simple producer hook: call this from your pipeline whenever you have composed frame and stats
def push_frame_bgr(frame_bgr, stats: dict):
    """Call this from your main pipeline. frame_bgr = BGR numpy image."""
    global LATEST_FRAME, LATEST_STATS
    _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with FRAME_LOCK:
        LATEST_FRAME = jpeg.tobytes()
        LATEST_STATS = stats.copy()

@app.route("/video_feed")
def video_feed():
    """MJPEG stream of latest frame."""
    def gen():
        global LATEST_FRAME
        while True:
            with FRAME_LOCK:
                f = LATEST_FRAME
            if f is None:
                # placeholder blank frame
                blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                _, jf = cv2.imencode(".jpg", blank)
                f = jf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
            time.sleep(0.04)  # ~25 FPS (adjust)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stats")
def stats():
    with FRAME_LOCK:
        return jsonify(LATEST_STATS)

@app.route("/alert", methods=["POST"])
def alert():
    """
    POST JSON: {"type": "sms"|"email", "to": "...", "message": "..."}
    Triggers alert via helper functions.
    """
    data = request.get_json(force=True)
    t = data.get("type")
    to = data.get("to")
    msg = data.get("message", "Alert from poultry monitor")
    if t == "sms":
        ok, res = send_sms_alert(to, msg)
    else:
        ok, res = send_email_alert(to, "Poultry Monitor Alert", msg)
    return jsonify({"ok": ok, "res": str(res)})

if __name__ == "__main__":
    # Example: start a small thread that simulates frames if you run standalone
    import numpy as np
    def fake_producer():
        while True:
            img = 255 * np.ones((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, time.strftime("%H:%M:%S"), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            stats = {"fps": 12.3, "num_tracked": 5, "health": "85%"}
            push_frame_bgr(img, stats)
            time.sleep(0.1)
    t = threading.Thread(target=fake_producer, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=8501, debug=False)
