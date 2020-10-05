"""
Classification of MNIST with scattering
=======================================
Here we demonstrate a simple application of scattering on the MNIST dataset.
We use 10000 images to train a linear classifier. Features are normalized by
batch normalization.
"""

###############################################################################
# Preliminaries

# from kymatio.torch import Scattering2D
# import torch
#
# scattering = Scattering2D(J=2, shape=(32, 32))
#
# #-----cpu-----
# x = torch.randn(1, 1, 32, 32)
# Sx = scattering(x)
#
# #-----gpu------
# scattering.cuda()
# x_gpu = x.cuda()
# Sx_gpu = scattering(x_gpu)
#
# Sx_gpu = Sx_gpu.cpu()
# print(torch.norm(Sx_gpu-Sx))
# scattering.cpu()


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from kymatio.torch import Scattering2D

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, original_data, title=None):
    digits = original_data
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig("./test.jpg")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.scattering = Scattering2D(J=2, shape=(28, 28))
        self.fc1 = nn.Linear(3969, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        bsz = x.shape[0]
        # print("flag1: ", x.shape)
        x = self.scattering(x)
        features = x.view(bsz, -1)
        # print("flag2: ", features.shape)
        x = self.fc1(features)
        output = F.log_softmax(x, dim=1)
        return output, features

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    all_features = []
    all_labels = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, features = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

        all_features.append(features.cpu())
        all_labels.append(target.cpu())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print("shape: ", all_features.shape, all_labels.shape)
    np.savez('./features/train.npz', all_features, all_labels)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_features = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, features = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_features.append(features.cpu())
            all_labels.append(target.cpu())
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print("shape: ", all_features.shape, all_labels.shape)
    np.savez('./features/test.npz', all_features, all_labels)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #------visualization--------
    # t-SNE embedding of the digits dataset
    # print("Computing t-SNE embedding")
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()

    # X_tsne = tsne.fit_transform(features.cpu())  # input is ndarray with shape [1083,64]  64 is the 8*8 dim.   1083 is the number of imgs
    # #X_tsne is [784,2]   target is 784 tensor
    #
    # original_data = data.squeeze()
    # plot_embedding(X_tsne, target.cpu().numpy(), original_data.cpu().numpy(),
    #                "t-SNE embedding of the digits (time %.2fs)" %
    #                (time() - t0))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # dataset1 = datasets.MNIST('../data', train=True, download=False,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)
    dataset1 = datasets.FashionMNIST('./data', train=True, download=False,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('./data', train=False,
                       transform=transform)


    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.train_batch_size, num_workers=1, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, num_workers=1, pin_memory=True, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "fashionmnist_scattering.pt")


if __name__ == '__main__':
    main()
