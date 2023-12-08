from Train2KD import *
print('=='*50,'\n','3-Train2KD Start','\n','=='*50)
trainer_KD2Train(trainer_path, trainer_dict)
model_Train2KD(model_path, model_dict)
loss_Train2KD(loss_path, loss_dict)
print('=='*50,'\n','3-Train2KD End','\n','=='*50)