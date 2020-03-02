"""
CNN model implementation to classify using a given HVR and a given taxonomic rank
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_preprocessing import test_loader_init, train_loader_init


# Class and functions
class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=7, stride=7)
        x = x.view(-1, 4 * 4 * 8)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(model, data_loader, loss_fn, optimizer, n_epochs=1):
    model.train(True)
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0

        for data in data_loader:
            inputs, labels = data
            bs = labels.size(0)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data.type(torch.LongTensor))
            running_loss += loss.data
            size += bs
        epoch_loss = running_loss.item() / size
        epoch_acc = running_corrects.item() / size
        loss_train[epoch_num] = epoch_loss
        acc_train[epoch_num] = epoch_acc
        print('Train - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        test(model, test_loader)
    return loss_train, acc_train


def test(model, data_loader):
    model.train(False)

    running_corrects = 0.0
    running_loss = 0.0
    size = 0

    for data in data_loader:
        inputs, labels = data

        bs = labels.size(0)
        # print(bs)

        outputs = model(inputs)
        # print(outputs)
        classes = labels == 1
        # print(classes)
        loss = loss_fn(outputs, classes.type(torch.LongTensor))
        _, preds = torch.max(outputs, 1)
        # print(preds)
        # print(classes)
        running_corrects += torch.sum(preds == classes.data.type(torch.LongTensor))
        running_loss += loss.data
        size += bs

    print('Test - Loss: {:.4f} Acc: {:.4f}'.format(running_loss / size, running_corrects.item() / size))


if __name__ == '__main__':
    train_loader = train_loader_init()
    test_loader = test_loader_init()
    conv_class = classifier()
    loss_fn = nn.NLLLoss()
    learning_rate = 1e-3
    optimizer_cl = torch.optim.Adam(conv_class.parameters(), lr=learning_rate)
