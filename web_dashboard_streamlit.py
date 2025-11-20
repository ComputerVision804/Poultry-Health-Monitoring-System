# web_dashboard_streamlit.py
# Requirements: streamlit, requests, PIL
# Run: streamlit run web_dashboard_streamlit.py
import streamlit as st, requests, time
from PIL import Image
from io import BytesIO

FLASK_BASE = st.sidebar.text_input("Flask base URL", "http://localhost:8501")
st.title("Poultry Monitor — Streamlit UI")

placeholder = st.empty()
st.sidebar.markdown("Controls")
refresh = st.sidebar.slider("Refresh (sec)", 0.1, 2.0, 0.5)

st.sidebar.markdown("Actions")
sms_to = st.sidebar.text_input("SMS to", "")
sms_msg = st.sidebar.text_area("SMS message", "Check flock — alert triggered")
if st.sidebar.button("Send SMS"):
    try:
        r = requests.post(f"{FLASK_BASE}/alert", json={"type":"sms","to":sms_to,"message":sms_msg}, timeout=5)
        st.sidebar.success(f"Sent: {r.json()}")
    except Exception as e:
        st.sidebar.error(str(e))

while True:
    try:
        # fetch image
        r = requests.get(f"{FLASK_BASE}/video_feed", stream=True, timeout=5)
        # video_feed is mjpeg; read a single frame
        bytes_buf = b''
        for chunk in r.iter_content(chunk_size=1024):
            bytes_buf += chunk
            if b'\xff\xd9' in bytes_buf:  # JPEG EOF
                start = bytes_buf.find(b'\xff\xd8')
                end = bytes_buf.find(b'\xff\xd9') + 2
                jpg = bytes_buf[start:end]
                bytes_buf = bytes_buf[end:]
                img = Image.open(BytesIO(jpg)).convert("RGB")
                placeholder.image(img, use_column_width=True)
                break

        s = requests.get(f"{FLASK_BASE}/stats", timeout=2)
        st.subheader("Stats")
        st.json(s.json())
    except Exception as e:
        st.error(f"Error: {e}")
    time.sleep(refresh)
