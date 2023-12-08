import sys
import os

from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect
print('=='*50,'\n','2-Prune Start','\n','=='*50)
def Prune(lastpath, savepath):
	"""没细读，能用就行"""
	# /media/cjz/Working/1-Code/5-Yolo8-Prune/runs/detect/d3f/weights/last.pt
	yolo = YOLO(lastpath)
	
	model = yolo.model
	ws,bs = [],[]
	for name, m in model.named_modules():
		if isinstance(m, torch.nn.BatchNorm2d):
			w = m.weight.abs().detach()
			b = m.bias.abs().detach()
			ws.append(w)
			bs.append(b)
	factor = 0.5
	ws = torch.cat(ws)
	threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
	def prune_conv(conv1: Conv, conv2: Conv):
		gamma = conv1.bn.weight.data.detach()
		beta = conv1.bn.bias.data.detach()
		keep_idxs = []
		local_threshold = threshold
		while len(keep_idxs) < 8:
			keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
			local_threshold = local_threshold * 0.5
		n = len(keep_idxs)
		conv1.bn.weight.data = gamma[keep_idxs]
		conv1.bn.bias.data = beta[keep_idxs]
		conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
		conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
		conv1.bn.num_features = n
		conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
		conv1.conv.out_channels = n
		if conv1.conv.bias is not None:
			conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]
		if not isinstance(conv2, list):
			conv2 = [conv2]
		for item in conv2:
			if item is not None:
				if isinstance(item, Conv):conv = item.conv
				else: conv = item
				conv.in_channels = n
				conv.weight.data = conv.weight.data[:, keep_idxs]
	def prune(m1, m2):
		if isinstance(m1, C2f): m1 = m1.cv2
		if not isinstance(m2, list):m2 = [m2]
		for i, item in enumerate(m2):
			if isinstance(item, C2f) or isinstance(item, SPPF):
				m2[i] = item.cv1
		prune_conv(m1, m2)
	for name, m in model.named_modules():
		if isinstance(m, Bottleneck):prune_conv(m.cv1, m.cv2)
	seq = model.model
	for i in range(3, 9):
		if i in [6, 4, 9]: continue
		prune(seq[i], seq[i + 1])
	detect: Detect = seq[-1]
	last_inputs = [seq[15], seq[18], seq[21]]
	colasts = [seq[16], seq[19], None]
	for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
		prune(last_input, [colast, cv2[0], cv3[0]])
		prune(cv2[0], cv2[1])
		prune(cv2[1], cv2[2])
		prune(cv3[0], cv3[1])
		prune(cv3[1], cv3[2])
	for name, p in yolo.model.named_parameters():p.requires_grad = True
	yolo.export(format="onnx") 
	torch.save(yolo.ckpt, savepath)
	print("prune done")


Prune('runs/detect/d3f/weights/last.pt',	# Projcet -> last.pt
	  'runs/detect/d3f/weights/prune.pt')	# save model
print('=='*50,'\n','2-Prune End','\n','=='*50)