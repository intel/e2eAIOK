import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import timm
import datetime
################# 1. Define Data Preprocessor #################
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343) # mean for 3 channels
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)  # std for 3 channels

train_transform = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

test_transform = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.ToTensor(),
  transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

################# 2. Prepare dataset and dataloader #################
batch_size = 128
num_workers = 1 # data worker
data_folder='./dataset' # dataset location
train_set = datasets.CIFAR100(root=data_folder, train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR100(root=data_folder, train=False, download=True, transform=test_transform)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
validate_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

################# 3. create underlying model #################
def initWeights(layer):
    ''' Initialize layer parameters

    :param layer: the layer to be initialized
    :return:
    '''
    classname = layer.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(layer.weight, 1.0, 0.02)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    else:
        print("no init layer [{}]".format(classname))
        # print("no init layer[%s]"%classname)        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('resnet50', pretrained=False, num_classes=100).to(device)
model.apply(initWeights) # Model Acquisition Is Initialization(MAII)
################# 4. create optimizer #################
init_lr = 0.01
weight_decay = 0.005
momentum = 0.9
optimizer = optim.SGD(model.parameters(),lr=init_lr, weight_decay=weight_decay,momentum=momentum)
################# 5. create scheduler #################
T_max = 100 # max 100 epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

################# 6. create trainer  #################
loss_fn = torch.nn.CrossEntropyLoss()
print_interval = 100 
def accuracy(output,label):
    pred = output.data.cpu().max(1)[1]
    label = label.data.cpu()
    if label.shape == output.shape:
        label = label.max(1)[1]
    return torch.mean((pred == label).float())

class Trainer:
    def __init__(self, model, optimizer, scheduler):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._best_acc = 0.0
        
    def train(self, train_dataloader, valid_dataloader, max_epoch):
        ''' 
        :param train_dataloader: train dataloader
        :param valid_dataloader: validation dataloader
        :param max_epoch: steps per epoch
        '''
        for epoch in range(0, max_epoch):
            ################## train #####################
            model.train()  # set training flag
            for (cur_step,(data, label)) in enumerate(train_dataloader):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss_value = loss_fn(output, label)
                loss_value.backward()       
                if cur_step%print_interval == 0:
                    batch_acc = accuracy(output,label)
                    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
                    print("[{}] epoch {} step {} : training batch loss {:.4f}, training batch acc {:.4f}".format(
                      dt, epoch, cur_step, loss_value.item(), batch_acc.item()))
                self._optimizer.step()
            self._scheduler.step()
            ################## evaluate ######################
            with torch.no_grad():
                model.eval()  
                loss_cum = 0.0
                sample_num = 0
                acc_cum = 0.0
                for (cur_step,(data, label)) in enumerate(valid_dataloader):
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    batch_size = data.size(0)
                    sample_num += batch_size
                    loss_cum += loss_fn(output, label).item() * batch_size
                    acc_cum += accuracy(output, label).item() * batch_size
                dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # date time
                if sample_num > 0:
                    loss_value = loss_cum/sample_num
                    acc_value = acc_cum/sample_num
                else:
                    loss_value = 0.0
                    acc_value = 0.0
                if acc_value >= self._best_acc:
                    self._best_acc = acc_value
                    torch.save(model.state_dict(), "./best_resnet50.pth")
                    print("Save model: epoch {}, best acc  {:.4f}".format(epoch,acc_value))
                    
                print("[{}] epoch {} : lr {}, evaluation loss {:.4f}, evaluation acc {:.4f}".format(
                    dt, epoch, self._scheduler.get_last_lr(),loss_value, acc_value))
trainer = Trainer(model, optimizer, scheduler)
################# 7. train #################
trainer.train(train_loader,validate_loader,T_max)