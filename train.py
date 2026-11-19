from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="road-damage.v2i.yolov8/data.yaml",
    epochs=10,
    imgsz=416,
    batch=2,
    workers=0,
    device="cpu",
    fraction=0.20,
    patience=5,
    project="runs",
    name="road_damage_v1"
)