from Train2KD import *
print('=='*50,'\n','0-Clear Start','\n','=='*50)
# clear
trainer_KD2Train(trainer_path, trainer_dict)
model_KD2Train(model_path, model_dict)
loss_KD2Train(loss_path, loss_dict)

# train
trainer_Train2KD(trainer_path, trainer_dict)
print('=='*50,'\n','0-Clear End','\n','=='*50)