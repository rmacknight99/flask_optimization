import shutil
from make_simulator import make_dataset, train_emulator

# required arguments #
try:
    shutil.move("data.csv", "datasets/dataset_custom/data.csv")
except:
    pass

name = 'custom'
batch_size = 50
reg = 0.001
hidden_act = 'leaky_relu'
out_act = 'relu'
max_epochs = 10000
feature_transform = 'normalize'
target_transform = 'normalize'

print('required arguments:\n')
print('name: {} (dataset name)'.format(name))
print('batch_size: {} (size of training point batch)'.format(batch_size))
print('hidden_act: {} (hidden activation function)'.format(hidden_act))
print('out_act: {} (output activation function)'.format(out_act))
print('max_epochs: {} (maximum number of training epochs)'.format(max_epochs))
print('feature_transform: {} (data transformation for features)'.format(feature_transform))
print('target_transform: {} (data transformation for targets)'.format(target_transform))


bnn,emulator,dataset,scores=train_emulator(name,batch_size,reg,hidden_act,out_act,max_epochs,feature_transform,target_transform,save=True) 
