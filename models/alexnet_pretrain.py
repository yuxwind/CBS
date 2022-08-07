# source: https://www.kaggle.com/drvaibhavkumar/alexnet-in-pytorch-cifar10-clas-83-test-accuracy

import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
import shutil
import os

#sys.path.append('..')
#from utils.checkpoints import save_checkpoint

def save_checkpoint(epoch, model_config, model, optimizer, lr_scheduler,
                    checkpoint_path: str, is_best=False, nick=''):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states.
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    if is_best:
        print(f"saving best checkpoint at epoch {epoch}")

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = {
        'type': type(optimizer),
        'state_dict': optimizer.state_dict()
    }
    checkpoint_dict['lr_scheduler'] = {
        'type': type(lr_scheduler),
        'state_dict': lr_scheduler.state_dict()
    }
    if nick == '':
        ckpt_suffix = ''
    else:
        ckpt_suffix = '_' + nick

    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{ckpt_suffix}.ckpt')
    path_best = os.path.join(checkpoint_path, f'best_checkpoint{ckpt_suffix}.ckpt')
    torch.save(checkpoint_dict, path_regular)
    print('====checkpoint_path', path_regular)
    if is_best:
        shutil.copyfile(path_regular, path_best)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

#Now using the AlexNet
AlexNet_Model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_Model.eval()
import torch.nn as nn
AlexNet_Model.classifier[1] = nn.Linear(9216,4096)
AlexNet_Model.classifier[4] = nn.Linear(4096,1024)
AlexNet_Model.classifier[6] = nn.Linear(1024,10)
AlexNet_Model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#AlexNet_Model = torch.nn.DataParallel(AlexNet_Model)
AlexNet_Model.to(device)

def test():
    #Testing Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = AlexNet_Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total
    #print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(AlexNet_Model.parameters(), lr=0.005, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

best_acc  = 100.0
model_config = {'arch': 'alexnet', 'dataset': 'cifar10', 'use_butterfly': False}
checkpoint_path = 'transfer_learning_alexnet_ckp'

for epoch in range(25):  # loop over the dataset multiple times
    optimizer.step()
    scheduler.step(epoch)
    print(f'epoch: {epoch}, LR: {scheduler.get_lr()}')
    running_loss = 0.0
    start_time = time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_Model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        #Time
        end_time = time.time()
        time_taken = end_time - start_time

        # print statistics
        running_loss += loss.item()
    test_acc = test()
    if test_acc < best_acc:
        save_checkpoint(epoch, model_config, AlexNet_Model, optimizer, scheduler, 
                checkpoint_path, is_best=True, nick='alexnet_transfer_learning')

        #if i % 2000 == 1999:    # print every 2000 mini-batches
    print('[%d, %5d] loss: %.6f, test_acc: %.4f %%' % (epoch + 1, i + 1, running_loss / i, test()*100))
    print('Time:',time_taken)
    running_loss = 0.0

print('Finished Training of AlexNet')

