"""
CNN model implementation to classify using a given HVR and a given taxonomic rank
"""
import numpy as np
import torch


# Running the train test
def train(model, train_loader, test_loader, loss_fn, optimizer, n_epochs=1):
    model.train(True)
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    loss_test = np.zeros(n_epochs)
    acc_test = np.zeros(n_epochs)
    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0

        for data in train_loader:
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
        epoch_loss_test, epoch_acc_test, _, _ = test(model, test_loader, loss_fn)
        loss_test[epoch_num] = epoch_loss_test
        acc_test[epoch_num] = epoch_acc_test
    return loss_train, acc_train, loss_test, acc_test


def test(model, test_loader, loss_fn):
    model.train(False)

    running_corrects = 0.0
    running_loss = 0.0
    size = 0

    running_labels = []
    running_preds = []

    for data in test_loader:
        inputs, labels = data

        bs = labels.size(0)

        outputs = model(inputs)
        running_labels = running_labels + list(labels)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_preds = running_preds + list(preds)
        running_corrects += torch.sum(preds == labels)
        running_loss += loss.data
        size += bs

    l_t = running_loss / size
    a_t = running_corrects.item() / size
    print('Test - Loss: {:.4f} Acc: {:.4f}'.format(running_loss / size, running_corrects.item() / size))
    return l_t, a_t, running_labels, running_preds
