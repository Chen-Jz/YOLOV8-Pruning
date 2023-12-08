import ultralytics

ultralytics_path = str(ultralytics.__file__).split('__init__.py')[0]
print('当前库位置 ultralytics：',ultralytics_path)

def trainer_Train2KD(trainer_path, trainer_dict):
	with open(trainer_path, 'r',errors='ignore') as f: train_py = f.read()
	train_py = train_py.replace(trainer_dict['train'], trainer_dict['KD'])
	with open(trainer_path, 'w') as f: f.write(train_py)
def trainer_KD2Train(trainer_path, trainer_dict):
	with open(trainer_path, 'r',errors='ignore') as f: train_py = f.read()
	train_py = train_py.replace(trainer_dict['KD'],trainer_dict['train'])
	with open(trainer_path, 'w') as f: f.write(train_py)

	
def model_Train2KD(model_path, model_dict):
	with open(model_path, 'r',errors='ignore') as f: model_py = f.read()
	model_py = model_py.replace(model_dict['train'][0], model_dict['KD'][0]).replace(model_dict['train'][1], model_dict['KD'][1])
	with open(model_path, 'w') as f: f.write(model_py)
def model_KD2Train(model_path, model_dict):
	with open(model_path, 'r',errors='ignore') as f: model_py = f.read()
	model_py = model_py.replace(model_dict['KD'][0],model_dict['train'][0]).replace(model_dict['KD'][1],model_dict['train'][1])
	with open(model_path, 'w') as f: f.write(model_py)


def loss_Train2KD(loss_path, loss_dict):
	with open(loss_path, 'r',errors='ignore') as f: loss_py = f.read()
	loss_py = loss_py.replace(loss_dict['train'], loss_dict['KD'])
	with open(loss_path, 'w') as f: f.write(loss_py)
def loss_KD2Train(loss_path, loss_dict):
	with open(loss_path, 'r',errors='ignore') as f: loss_py = f.read()
	loss_py = loss_py.replace(loss_dict['KD'],loss_dict['train'])
	with open(loss_path, 'w') as f: f.write(loss_py)




# ============== 需要修改部分 ============== #
trainer_path = f'{ultralytics_path}engine/trainer.py'
model_path = f'{ultralytics_path}engine/model.py'
loss_path = f'{ultralytics_path}utils/loss.py'
# ============== 需要修改部分 ============== #







trainer_dict = {
'train' : 'self.scaler.scale(self.loss).backward()',
'KD' : '''self.scaler.scale(self.loss).backward()
                l1_lambda = 1e-2 * (1 - 0.9 * epoch / self.epochs)
                for k, m in self.model.named_modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.grad.data.add_(l1_lambda * torch.sign(m.weight.data))
                        m.bias.grad.data.add_(1e-2 * torch.sign(m.bias.data))\n'''}
model_dict = {
'train':['self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)','self.model = self.trainer.model'],
'KD' : ['# KD #', 'self.trainer.model = self.model.train()']
}
loss_dict = {
'train':'b, a, c = pred_dist.shape',
'KD':'''device = pred_dist.device
            self.proj = self.proj.to(device)
            b, a, c = pred_dist.shape  # batch, anchors, channels\n
'''
}