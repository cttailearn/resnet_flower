'''
CNN网络识别花朵,使用resnet模型
'''
import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets,models


# 数据路径
data_dir = 'flower_data/'
train_dir = data_dir + '/train'
val_dir = data_dir + '/valid'

#图像预处理操作
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([96,96]),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
       transforms.Resize([96,96]),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#数据源
batch_size = 64
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

#读取花朵标签对应名称
with open('flower_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

num_classes = len(flower_to_name)

#加载网络模型
model_name= 'resnet'
feature_extract = False

#是否使用gpu训练
def train_on_gup():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')    
    else: 
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#模型输出层需要根据自己的标签来修改
def initialize_model(model_name, num_classes, feature_extract, model_file=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 64

    if model_file == "None":
        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(weights="IMAGENET1K_V1")
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        checkpoint = torch.load(model_file)
        model_ft.load_state_dict(checkpoint['model_state_dict'])

    return model_ft, input_size


#训练
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, file_name='model.pth'):
    since = time.time()
    best_acc = 0
    device = train_on_gup()
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    lrs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            runing_loss = 0.0
            runing_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 输入数据
                labels = labels.to(device)  # 输入标签

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)    
                if phase == 'train':
                    loss.backward()
                    optimizer.step()    

                runing_loss += loss.item() * inputs.size(0) 
                runing_corrects += torch.sum(preds == labels.data)  
        
            epoch_loss = runing_loss / len(dataloaders[phase].dataset)  
            epoch_acc = runing_corrects.double() / len(dataloaders[phase].dataset)
            time_elapsed = time.time() - since
            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {'epoch': epoch, 'best_acc':best_acc, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss}
                torch.save(state, file_name)
            if phase == 'valid':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history, lrs

if __name__ == '__main__':
    file_name='flower_resnet18.pth'
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, model_file=file_name)
    #优化器
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)#学习率每5个epoch衰减成原来的1/10
    criterion = nn.CrossEntropyLoss()
    model_ft,val_acc_history,train_acc_history,train_loss_history,val_loss_history,lrs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=10,file_name=file_name)