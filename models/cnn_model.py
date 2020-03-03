"""
CNN model implementation to classify using a given HVR and a given taxonomic rank
"""
import torch.nn as nn
import torch.nn.functional as F

# Class and functions
"""
class classifier_basic(nn.Module):
    # Observed accuracy on test set with V4 and taxonomy level 1 is around 33%

    def __init__(self, n_out_features: int):
        super(classifier_basic, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, padding=0)
        self.fc = nn.Linear(in_features=8 * ((300 - 4 + 1) // 4), out_features=n_out_features)

    def forward(self, x):
        # x.size() = [64, 4, 300]
        x = self.conv1(x)
        # x.size() = [64, 8, 297]
        x = F.max_pool1d(x, kernel_size=4, stride=4)
        # x.size() = [64, 8, 74]
        x = x.view(-1, 8 * 74)
        # x.size() = [64, 592]
        x = self.fc(x)
        # x.size() = [64, 226]
        return F.log_softmax(x, dim=1)


class classifier_basic_2(nn.Module):
    # Observed accuracy on test set with V4 and taxonomy level 1 is around 33% too

    def __init__(self, n_out_features: int):
        super(classifier_basic_2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, padding=0)
        self.fc = nn.Linear(in_features=8 * (300 - 4 + 1), out_features=n_out_features)

    def forward(self, x):
        # x.size() = [64, 4, 300]
        x = self.conv1(x)
        # x.size() = [64, 8, 297] - Removing max_pool1d
        # x = F.max_pool1d(x, kernel_size=4, stride=4)
        # x.size() = [64, 8, 297]
        x = x.view(-1, 8 * 297)
        # x.size() = [64, 8 * 297]
        x = self.fc(x)
        # x.size() = [64, 226]
        return F.log_softmax(x, dim=1)


# We observe that those two classifiers overfit largely the train set, let's give less parameters to optimize

class classifier_less_parameters(nn.Module):
    # Observed accuracy on test set with V4 and taxonomy level 1 is around 33% too

    def __init__(self, n_out_features: int):
        super(classifier_less_parameters, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=4, padding=0)
        self.fc = nn.Linear(in_features=4 * ((300 - 4 + 1) // 8), out_features=n_out_features)

    def forward(self, x):
        # x.size() = [64, 4, 300]
        x = self.conv1(x)
        # x.size() = [64, 4, 297]
        x = F.max_pool1d(x, kernel_size=4, stride=8)
        # x.size() = [64, 4, 297//8] = [64, 8, 37]
        x = x.view(-1, 4 * 37)
        # x.size() = [64, 4 * 37]
        x = self.fc(x)
        # x.size() = [64, 148]
        return F.log_softmax(x, dim=1)


# We observe that those three classifiers overfit largely the train set, let's test models from
# "Convolutional neural networks for classification of alignments of non-coding RNA sequences" paper

class classifier_Aoki_1(nn.Module):
    # Observed accuracy on test set with V4 and taxonomy level 1 is around 35% with minimal parameters

    def __init__(self, n_out_features: int):
        super(classifier_Aoki_1, self).__init__()
        # Parameters
        self.out_channel_1 = 6
        self.kernel_size_1 = 3
        self.max_pool_stride_1 = 8
        self.L_out_1 = int((300 - self.kernel_size_1) // self.max_pool_stride_1 + 1)
        # Layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel_1,
                               kernel_size=self.kernel_size_1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.out_channel_1)
        self.ReLU1 = nn.ReLU()
        # Parameters
        self.out_channel_2 = 13
        self.kernel_size_2 = self.kernel_size_1
        self.max_pool_stride_2 = 8
        self.L_out_2 = int((self.L_out_1 - self.kernel_size_2) // self.max_pool_stride_2 + 1)
        # Layers
        self.conv2 = nn.Conv1d(in_channels=self.out_channel_1, out_channels=self.out_channel_2,
                               kernel_size=self.kernel_size_2, padding=0)
        self.bn2 = nn.BatchNorm1d(self.out_channel_2)
        self.ReLU2 = nn.ReLU()

        # Hidden part
        self.ratio_fc_1 = 1 / 2
        self.out_fc_1 = int(self.out_channel_2 * self.L_out_2 * self.ratio_fc_1)
        self.fc1 = nn.Linear(in_features=self.out_channel_2 * self.L_out_2,
                             out_features=self.out_fc_1)
        self.ReLU3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.out_fc_1,
                             out_features=n_out_features)

    def forward(self, x):
        # CONVOLUTION 1
        # x.size() = [64, 4, 300]
        x = self.conv1(x)
        # # print(x.size())
        # x.size() = [64, out_chanel_1, 300-kernel_size_1+1]
        # Add a trainable BatchNormalization and a simple RELU layer
        x = self.bn1(x)
        x = self.ReLU1(x)
        # x.size() = same
        x = F.max_pool1d(x, kernel_size=self.kernel_size_1, stride=self.max_pool_stride_1)
        # # print(x.size())
        # x.size() = [64, out_chanel_1, L_out_1 = (300-kernel_size_1+1)//max_pool_stride_1]

        # CONVOLUTION 2
        # x.size() = [64, out_chanel_1, L_out_1 = (300-kernel_size_1+1)//max_pool_stride_1]
        x = self.conv2(x)
        # # print(x.size())
        # x.size() = [64, out_chanel_2, (L_out_1-kernel_size_2+1)]
        # Add a trainable BatchNormalization and a simple RELU layer
        x = self.bn2(x)
        x = self.ReLU2(x)
        # x.size() = same
        x = F.max_pool1d(x, kernel_size=self.kernel_size_2, stride=self.max_pool_stride_2)
        # # print(x.size())
        # x.size() = [64, out_chanel_2, L_out_2 = int((L_out_1-kernel_size_2+1)//max_pool_stride_2)]

        # Hidden layers
        # A first fully connected one
        x = x.view(-1, self.out_channel_2 * self.L_out_2)
        # # print(x.size())
        x = self.fc1(x)
        # # print(x.size())
        # x.size() = [64, self.out_fc_1 = int(self.out_channel_2 * self.L_out_2 * self.ratio_fc_1)]
        # RELU
        x = self.ReLU3(x)
        # Add a dropout layer to prevent co-adaptation of neurons
        x = F.dropout(x, p=0.5)
        # x.size() = [64, self.out_fc_1]
        x = self.fc2(x)
        # # print(x.size())
        # x.size() = [64, n_out_features]
        return F.log_softmax(x, dim=1)


class classifier_Aoki_2(nn.Module):
    # Observed accuracy on test set with V4 and taxonomy level 1 is around 35% with second minimal parameters
    # Observed accuracy on test set with V3 and taxonomy level 1 is around 25% with second minimal parameters
    # Observed accuracy on test set with V4 and taxonomy level 0 is 100% with second minimal parameters

    def __init__(self, n_out_features: int):
        super(classifier_Aoki_2, self).__init__()
        # Parameters
        self.out_channel_1 = 13
        self.kernel_size_1 = 3
        self.max_pool_stride_1 = 8
        self.L_out_1 = int((300 - self.kernel_size_1) // self.max_pool_stride_1 + 1)
        # Layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel_1,
                               kernel_size=self.kernel_size_1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.out_channel_1)
        self.ReLU1 = nn.ReLU()
        # Parameters
        self.out_channel_2 = 26
        self.kernel_size_2 = self.kernel_size_1
        self.max_pool_stride_2 = 8
        self.L_out_2 = int((self.L_out_1 - self.kernel_size_2) // self.max_pool_stride_2 + 1)
        # Layers
        self.conv2 = nn.Conv1d(in_channels=self.out_channel_1, out_channels=self.out_channel_2,
                               kernel_size=self.kernel_size_2, padding=0)
        self.bn2 = nn.BatchNorm1d(self.out_channel_2)
        self.ReLU2 = nn.ReLU()

        # Hidden part
        self.ratio_fc_1 = 3 / 4
        self.out_fc_1 = int(self.out_channel_2 * self.L_out_2 * self.ratio_fc_1)
        self.fc1 = nn.Linear(in_features=self.out_channel_2 * self.L_out_2,
                             out_features=self.out_fc_1)
        self.ReLU3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.out_fc_1,
                             out_features=n_out_features)

    def forward(self, x):
        # CONVOLUTION 1
        # x.size() = [64, 4, 300]
        x = self.conv1(x)
        # # print(x.size())
        # x.size() = [64, out_chanel_1, 300-kernel_size_1+1]
        # Add a trainable BatchNormalization and a simple RELU layer
        x = self.bn1(x)
        x = self.ReLU1(x)
        # x.size() = same
        x = F.max_pool1d(x, kernel_size=self.kernel_size_1, stride=self.max_pool_stride_1)
        # # print(x.size())
        # x.size() = [64, out_chanel_1, L_out_1 = (300-kernel_size_1+1)//max_pool_stride_1]

        # CONVOLUTION 2
        # x.size() = [64, out_chanel_1, L_out_1 = (300-kernel_size_1+1)//max_pool_stride_1]
        x = self.conv2(x)
        # # print(x.size())
        # x.size() = [64, out_chanel_2, (L_out_1-kernel_size_2+1)]
        # Add a trainable BatchNormalization and a simple RELU layer
        x = self.bn2(x)
        x = self.ReLU2(x)
        # x.size() = same
        x = F.max_pool1d(x, kernel_size=self.kernel_size_2, stride=self.max_pool_stride_2)
        # # print(x.size())
        # x.size() = [64, out_chanel_2, L_out_2 = int((L_out_1-kernel_size_2+1)//max_pool_stride_2)]

        # Hidden layers
        # A first fully connected one
        x = x.view(-1, self.out_channel_2 * self.L_out_max_pool_2)
        # # print(x.size())
        x = self.fc1(x)
        # # print(x.size())
        # x.size() = [64, self.out_fc_1 = int(self.out_channel_2 * self.L_out_2 * self.ratio_fc_1)]
        # RELU
        x = self.ReLU3(x)
        # Add a dropout layer to prevent co-adaptation of neurons
        x = F.dropout(x, p=0.5)
        # x.size() = [64, self.out_fc_1]
        x = self.fc2(x)
        # # print(x.size())
        # x.size() = [64, n_out_features]
        return F.log_softmax(x, dim=1)
"""


class classifier_GD_1(nn.Module):
    # Observed accuracy on test set with V4 and taxonomy level 1 is around 35% with these parameters

    def __init__(self, n_out_features: int):
        super(classifier_GD_1, self).__init__()
        # PARAMETERS
        self.out_channel_1 = 30
        self.out_channel_2 = 30
        self.kernel_size_1 = 4
        self.max_pool_stride_1 = 8
        self.max_pool_stride_2 = 8
        self.ratio_fc_1 = 1 / 2
        # COPIED PARAMETERS
        self.kernel_size_max_pool_1 = self.kernel_size_1
        self.kernel_size_2 = self.kernel_size_1
        # SIZE COMPUTATION
        self.L_out_conv_1 = 300 - self.kernel_size_1 + 1
        self.L_out_max_pool_1 = int((self.L_out_conv_1 - self.kernel_size_1) // self.max_pool_stride_1) + 1
        self.L_out_conv_2 = self.L_out_max_pool_1 - self.kernel_size_2 + 1
        self.L_out_max_pool_2 = int((self.L_out_conv_2 - self.kernel_size_2) // self.max_pool_stride_2) + 1
        self.L_out_fc_1 = int(self.out_channel_2 * self.L_out_max_pool_2 * self.ratio_fc_1)

        # Layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.out_channel_1,
                               kernel_size=self.kernel_size_1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.out_channel_1)
        self.ReLU1 = nn.ReLU()
        # Layers
        self.conv2 = nn.Conv1d(in_channels=self.out_channel_1, out_channels=self.out_channel_2,
                               kernel_size=self.kernel_size_2, padding=0)
        self.bn2 = nn.BatchNorm1d(self.out_channel_2)
        self.ReLU2 = nn.ReLU()

        # Hidden part
        self.fc1 = nn.Linear(in_features=self.out_channel_2 * self.L_out_max_pool_2,
                             out_features=self.L_out_fc_1)
        self.ReLU3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.L_out_fc_1,
                             out_features=n_out_features)

    def forward(self, x):
        # CONVOLUTION 1
        # x.size() = [64, 4, 300]
        x = self.conv1(x)
        # print(x.size())
        # x.size() = [64, out_chanel_1, L_out_conv_1 = 300-kernel_size_1+1]
        # Add a trainable BatchNormalization and a simple RELU layer
        x = self.bn1(x)
        x = self.ReLU1(x)
        # x.size() = same
        x = F.max_pool1d(x, kernel_size=self.kernel_size_1, stride=self.max_pool_stride_1)
        # print(x.size())
        # x.size() = [64, out_chanel_1, L_out_max_pool_1 = (L_out_conv_1-kernel_size_1)//max_pool_stride_1 + 1]

        # CONVOLUTION 2
        x = self.conv2(x)
        # # print(x.size())
        # x.size() = [64, out_chanel_2, L_out_conv_2 = (L_out_max_pool_1-kernel_size_2+1)]
        # Add a trainable BatchNormalization and a simple RELU layer
        x = self.bn2(x)
        x = self.ReLU2(x)
        # x.size() = same
        x = F.max_pool1d(x, kernel_size=self.kernel_size_2, stride=self.max_pool_stride_2)
        # print(x.size())
        # x.size() = [64, out_chanel_2, L_out_max_pool_2 = int((L_out_conv_2-kernel_size_2)//max_pool_stride_2)+1]

        # Hidden layers
        # A first fully connected one
        x = x.view(-1, self.out_channel_2 * self.L_out_max_pool_2)
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        # x.size() = [64, self.L_out_fc_1 = int(self.out_channel_2 * self.L_out_max_pool_2 * self.ratio_fc_1)]
        # RELU
        x = self.ReLU3(x)
        # Add a dropout layer to prevent co-adaptation of neurons
        x = F.dropout(x, p=0.5)
        # x.size() = [64, self.L_out_fc_1]
        x = self.fc2(x)
        # print(x.size())
        # x.size() = [64, n_out_features]
        return x  # With CrossEntropyLoss directly