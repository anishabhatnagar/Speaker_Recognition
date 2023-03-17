####Resnet 34 150 epochs####
print("importing modules")
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFile
from torchvision import models

import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import time
import copy
print("modules imported")
# to load any image having errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("\n")
#Reading the csv file as dataframe
print("reading csv  file")
data=pd.read_csv("DataFrame.csv")

#Encoding the labels
print("encoding labels")
lb = LabelEncoder()
data['encoded_labels'] = lb.fit_transform(data['Label'])

print("\n")
print("DATA")
#print (data.head())
print (data[1:30])

print("\n")
#Number of speakers
speakers=data['encoded_labels'].unique()
speaker_count=len(speakers)
print("number of speaker: ",speaker_count)


#Number of Voice samples
samples= len(data['Mel_DB_spec'])
print ("number of samples in the dataset: ", samples)

#Pre-processing
#transform the images
input_size = 224
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(input_size),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Spectogram_Dataset(Dataset):
    def __init__(self, img_data,img_path,transform=None):
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        # img_name = os.path.join(self.img_path,self.img_data.loc[index, 'labels'],
        #                         self.img_data.loc[index, 'Images'])
        img_name =self.img_data.loc[index, 'Mel_DB_spec']
        img_name = str(img_name) + ".png"
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = image.resize((300,300))
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

print("\n")
BASE_PATH= "/home/ubuntu/GRAPHS/melDB"
print("BASE_PATH",BASE_PATH)
print("applying transforms on dataset")
dataset = Spectogram_Dataset(data,BASE_PATH,transform)


print("\n")
#Splitting the dataset
batch_size = 64 
print("batch_size = ",batch_size)
shuffle_dataset = True
print("shuffle_dataset = ",shuffle_dataset)
random_seed= 42
print("random_seed = ",random_seed)
split=[0.8,0.5]
print("split index = ",split)




dataset_size = len(data)
print("\ndataset_size = ",dataset_size)
print("\nSPLITTING SET")
indices = list(range(dataset_size))

split_1 = int(np.floor(split[0] * dataset_size))
#print(split_1)
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, remaining_indices = indices[:split_1], indices[split_1:]

remaining_dataset_size= int(dataset_size-len(train_indices))
#print(remaining_dataset_size)
split_2= int(np.floor(split[1] * remaining_dataset_size))
#print(split_2)
test_indices, val_indices = remaining_indices[split_2:], remaining_indices[:split_2]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

print("training set size: ")
print(len(train_indices))
print("validation set size: ")
print(len(val_indices))
print("testing set size: ")
print(len(test_indices))


print("\ncreating DataLoaders")
#Creating Dataloaders
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler, drop_last=True)

dataloaders={"train":train_loader,"val":validation_loader}

print("DataLoaders created")

print("viewing samples")
#Display random images
def img_display(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# Viewing data examples used for training
fig, axis = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = images[i], labels[i]
        ax.imshow(img_display(image)) # add image
        ax.set(title = f"{label.item()}") # add label


print("defining model")
#Model Defination
num_classes =speaker_count
model = models.resnet18(pretrained=True)
print("model = resnet18")
for param in model.parameters():
    param.requires_grad =False
print("set requires grad =false")
print("changing last fc layer")
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, num_classes)
)

#print("\nmodel summary",summary(model,(3,224,224)))


print("\nSetting model parameters")
#Setting parameters
def set_lr_epochs(lr,epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    num_epochs= epochs
    return criterion,optimizer,lr,num_epochs

criterion,optimizer,lr,num_epochs=set_lr_epochs(1e-3,10)
print("criterion = ",criterion,"\noptimizer = ",optimizer,"\nlr = ",lr,"\nepochs = ",num_epochs)

print("model summary after freezing",summary(model,(3,224,224)))



def train_model_1():
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                 accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device),
                                      labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals =
                            top_class == labels.view(*top_class.shape)
                        accuracy +=
                        torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
model.train()
#torch.save(model, 'aerialmodel.pth')

#Testing and Training
train_loss=[]
val_loss=[]
print("training")
def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []

    #A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    num_epochs= num_epochs
    #increase the added value by 25 after every training run


    #change the starting value after every training run
    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
            if phase =='train':
                train_loss.append(epoch_loss)
            if phase == 'val':
                val_loss.append(epoch_loss)
        if epoch%10==0:
            PATH=os.path.join("/home/ubuntu/"+ "Middle_models/" + "resnet18_10ep" + str(epoch))
            torch.save(model.state_dict(), PATH)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

print("testing")
def test_model(test_loader,model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)


          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), PATH + 'test_resnet18_10ep.ckpt')

    #p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
    #p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    #p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    #p.line(np.arange(len(loss_list)), loss_list)
    #p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    #show(p)

#Shifting to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice",device)

print("\nshift model to cuda")
model = model.to(device)


#Loding pevious model
#print("loading previous model weights")
#model.load_state_dict(torch.load("model_resnet34_200-250_epochs"))


#TRAINING
print("calling training module")
train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False)


#Saving the trained model
print("saving trained model")
PATH=os.path.join("resnet18_10ep")
print("at ",PATH)
#torch.save(model.state_dict(), PATH)

epochs = range(1,11)
plt.plot(epochs, train_loss, 'g', label='Training accuracy')
plt.plot(epochs, val_loss, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#TESTING
print("calling testing module")
model.to(device)
test_model(test_loader,model)

print("TESTING OVERFITTING on train loader")
print("accuracy on train dataloader")
test_model(train_loader,model)

print("testing overfitting on validation loader")
print("accuracy on validation dataloader")
test_model(val_loader,model)


print("\nRESNET.py Finished")


