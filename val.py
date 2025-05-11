import warnings
warnings.filterwarnings('ignore')
import torch 
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/projects/WormReIDTracker-all/runs/train/exp78/weights/best.pt')

    results = model.val(
        data=r'coco8-deep-worm.yaml',
        split='val',
        imgsz=1024,
        batch=8,
        project='runs/val',
        name='exp',
    )
    
    print(f"box-mAP50: {results.box.map50}")  # map50
    print(f"box-mAP75: {results.box.map75}")  # map75
    print(f"box-mAP50-95: {results.box.map}")  # map50-95  
    speed_results = results.speed
    total_time = sum(speed_results.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}")
