"""
functions to create activation map from a given saved model - Based on Marc Lelarge Course notebook
"""
# Packages
import os
import random

import cv2
import numpy as np
import torch
import time
from torch.autograd import Variable
from torch.nn import functional as F

from models.cnn_model import classifier_GD_2_ACM
from models.cnn_preprocessing import get_homogenous_vector


def create_activation_map(X_test, y_test, dict_id_to_class, parameter_config, n: int = 5,
                          analysis_path: str = ''):
    """
    """
    n_out_features = len(dict_id_to_class)
    H_init_dim = 4 ** parameter_config['k_mer']
    W_init_dim = parameter_config['vector_max_size']
    max_size = parameter_config['vector_max_size']

    X_test_col = X_test.iloc[:, 1]
    y_test_col = y_test.iloc[:, 1]
    new_X_test = np.array([get_homogenous_vector(X_test_col[i], max_size).transpose() for i in range(len(X_test))])
    classes = dict_id_to_class

    conv_class = classifier_GD_2_ACM(n_out_features=n_out_features,
                                     parameter_config=parameter_config)

    model_path = analysis_path + 'model.pt'
    acm_path = analysis_path + 'ACM\\'

    if not os.path.exists(acm_path):
        os.makedirs(acm_path)

    conv_class.load_state_dict(torch.load(model_path))
    conv_class.eval()
    finalconv_name = 'conv'
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    conv_class._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(conv_class.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (H_init_dim, W_init_dim)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    for acm_number in range(n):
        index = random.choice(range(len(y_test_col)))

        seq = X_test_col[index]
        X_sample = torch.from_numpy(new_X_test[index].astype(np.float32))
        y_sample = y_test_col[index]
        seq_variable = Variable(X_sample.unsqueeze(0))
        logit = conv_class(seq_variable)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        y_pred = classes[idx[0]]

        # output the prediction
        print('Real class: {}'.format(y_sample))
        for i in range(0, 2):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # Create final image
        img = image_from_sequence(seq, max_size=max_size)
        height, width = 50, max_size * 50
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(acm_path + 'Sample_{}_Predicted_{}_{:.0f}.jpg'.format(y_sample, y_pred, time.time()*10), result)

    return


def image_from_sequence(seq, max_size: int = 300):
    image_sequence = np.zeros((50, 0, 3))
    for index in range(0, max_size):
        if index < len(seq):
            bw_img = cv2.imread(
                "utils/letters/icons8-{}-50.png".format(seq[index].lower()),
                cv2.IMREAD_UNCHANGED
            )[:, :, 3]
            letter_img = np.zeros((50, 50, 3))
            letter_img[:, :, 0] = 255 - bw_img
            letter_img[:, :, 1] = 255 - bw_img
            letter_img[:, :, 2] = 255 - bw_img
        else:
            letter_img = np.zeros((50, 50, 3)) + 255
        image_sequence = np.concatenate((image_sequence, letter_img), axis=1)

    return image_sequence
