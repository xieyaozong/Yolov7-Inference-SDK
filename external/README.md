# External YOLOv7 Checkout

This project no longer vendors the full YOLOv7 implementation.

Clone YOLOv7 outside this repository:

```powershell
git clone https://github.com/WongKinYiu/yolov7 C:\work\external\yolov7
```

Then run:

```powershell
python scripts/detect_image.py --image images\bus.jpg --weights models\yolov7.pt --yolov7-repo C:\work\external\yolov7
```

You can also set:

```powershell
$env:YOLOV7_REPO = "C:\work\external\yolov7"
```

