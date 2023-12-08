from ultralytics import YOLO
from Train2KD import *
print('*'*20,'Finetuning','*'*20)
model = YOLO('runs/detect/d3f/weights/prune.pt')
# 使用模型
model.train(data = "d3f.yaml", 
			epochs = 1,
			batch = 64,
			imgsz = 640,
			device = '7',
			workers = 16,
			name = f'd3f_finetun',
			amp=True
			)  

# clear
trainer_KD2Train(trainer_path, trainer_dict)
model_KD2Train(model_path, model_dict)
loss_KD2Train(loss_path, loss_dict)