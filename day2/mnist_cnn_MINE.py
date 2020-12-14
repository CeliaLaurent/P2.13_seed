from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    #define the building blocks that i need:
    def __init__(self):
        super(Net, self).__init__()
        #convolutional layers for 1 channel, 20 filters of size 5x5:
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # keep last parameter =1
        
        #another convolutional layer with an input of 20 channels 
        #(20 versions of the image given by the 20 filter)
        # this convolutiona layer has 50 filters of size 5x5:
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        
        # 500 is the width of the last hidden layer
        self.fc1 = nn.Linear(4*4*50, 500)
        # transformation from the size of the last
        # hidden layer to the output
        self.fc2 = nn.Linear(500, 10)

    # describe the flow of information inside the network:
    def forward(self, x):
        # do a convulution and apply the relu :
        x = F.relu(self.conv1(x))
        # take the max value of the maps  reducing the 
        # representation by a factor of 2x2=4 :
        x = F.max_pool2d(x, 2, 2)
        # again, do a convulution and apply the relu :
        x = F.relu(self.conv2(x))
        # further reducing the dimensionality of the reppresentation :
        x = F.max_pool2d(x, 2, 2)
        # for each datapoint a want one vector of length 4*4*50 :
        x = x.view(-1, 4*4*50)
        # flatten it into a single vector : 
        x = F.relu(self.fc1(x))
        # perform the last linear transformation applying the last
        # non-linearity :
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # for each Epoch iterate on the batches of the dataset :
    for batch_idx, (data, target) in enumerate(train_loader):
        # taking data
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #compute the forward path
        output = model(data)
        # compute the loss :
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default=False)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
