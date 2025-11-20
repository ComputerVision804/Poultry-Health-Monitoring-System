try:
    from ultralytics import YOLO
except:
    YOLO = None
    print("Install YOLO: pip install ultralytics")
