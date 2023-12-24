import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy  # Import the copy module


import torchvision
from torchvision import transforms
import utils
from utils import prune_network,feature_selection_function,Variance_,feature_selection_norme


def trouver_parmetre_variance(net,delte):
    while(utils.evaluate(net, testloader, device)>0.8):
       net = prune_network(net,Variance_ , None, delte)
       delte=delte+0.0001
    return net

def trouver_parmetre_l1(net,delte):
    while(utils.evaluate(net, testloader, device) >0.9 ):
       net = prune_network(net, feature_selection_norme, None, delte)
       delte=delte+1
    return net


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    # print(trainloader)
    net = LeNet()

# Élaguer le réseau
    etat_modele = torch.load('./lenet_mnist.pth', map_location=torch.device('cpu'))
    net.load_state_dict(etat_modele)
    utils.evaluate(net, testloader, device)
    # trouver_parmetre_variance(net,0.00002)
    trouver_parmetre_l1(net,1)
    # pruned_model = prune_network(net, feature_selection_function, None, 0.003)
    # pruned_model(next(iter(trainloader))) //2
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # utils.train(net, trainloader, criterion, optimizer, device, 10)
    # utils.evaluate(pruned_model, testloader, device)


    # torch.save(net.state_dict(), '../models/lenet_mnist.pth')


        



