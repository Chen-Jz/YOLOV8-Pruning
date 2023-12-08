import sys
import os

from ultralytics import YOLO


print('=='*50,'\n','1-Train Start','\n','=='*50)
Train_model = YOLO('yolov8n.pt')        	# 选择模型大小
Train_model.train(data = "d3f.yaml",		# 选择数据data配置
			epochs = 1,
			batch = 192,
			imgsz = 640,
			device = '4,5,6,7',				# GPU ID
			workers = 8,					
			name = 'd3f',					# Project
			amp=False, 						# 一定False
			exist_ok = True)

print('=='*50,'\n','1-Train End','\n','=='*50)