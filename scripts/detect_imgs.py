from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # yolo11s.pt
for r in model("path/to/your/image.jpg"):  # Replace with the path to your image
    r.show()  # Display the results
