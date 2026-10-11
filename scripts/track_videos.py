from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # yolo11s.pt
model.track(
    source="path/to/your/video.mp4",
    tracker="bytetrack.yaml",  # botsort.yaml
    classes=[0, 2],
    save=True,
    show=False,
)  # Replace with the path to your video
