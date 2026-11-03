import subprocess, sys, os, torch

print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'],
                   capture_output=True, text=True)
print('pip ultralytics:', r.returncode, (r.stderr or '')[-300:])

from ultralytics import YOLO
model = YOLO('yolov8n.pt')
res = model.train(
    data='/content/dataset_47n/data.yaml',
    epochs=60, imgsz=640, batch=16, device=0, workers=4,
    patience=20, lr0=0.01,
    project='/content/runs', name='oilpalm47n', exist_ok=True,
)
print('SELESAI')
best = '/content/runs/oilpalm47n/weights/best.pt'
print('best.pt ada:', os.path.exists(best), os.path.getsize(best) if os.path.exists(best) else '-')
