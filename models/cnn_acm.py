"""
functions to create activation map from a given saved model - Based on Marc Lelarge Course notebook
"""
# Packages
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from models.cnn_model import classifier_GD_2_ACM
from models.cnn_preprocessing import get_homogenous_vector
from utils.utils import slash


# MAIN
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
    acm_path = analysis_path + 'ACM{}'.format(slash)

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
        # print('Real class: {}'.format(y_sample))
        # for i in range(0, 2):
        #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # Create final image
        img = image_from_sequence(seq, max_size=max_size)
        height, width = 50, max_size * 50
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite(acm_path + 'Sample_{}_Predicted_{}_{:.0f}.png'.format(y_sample, y_pred, time.time() * 10), result)

    return


# SUB MAIN
def create_one_activation_map_with_return(atcg_seq, new_X_seq, y_test_col, test_id, dict_id_to_class, parameter_config,
                                          analysis_path, max_size=300):
    """
    """
    H_init_dim = 4 ** parameter_config['k_mer']
    W_init_dim = max_size

    classes = dict_id_to_class

    finalconv_name = 'conv'
    features_blobs = []

    n_out_features = len(dict_id_to_class)

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    conv_class = classifier_GD_2_ACM(n_out_features=n_out_features,
                                     parameter_config=parameter_config)

    model_path = analysis_path + 'model.pt'

    conv_class.load_state_dict(torch.load(model_path))
    conv_class.eval()
    finalconv_name = 'conv'
    features_blobs = []

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

    index = test_id

    seq = atcg_seq[index]
    X_sample = torch.from_numpy(new_X_seq[index].astype(np.float32))
    y_sample = y_test_col[index]
    seq_variable = Variable(X_sample.unsqueeze(0))
    logit = conv_class(seq_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    y_pred = classes[idx[0]]

    # output the prediction
    # print('Real class: {}'.format(y_sample))
    # for i in range(0, 2):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # Create final image
    img = cv2.resize(image_from_sequence(seq, max_size=max_size), (10 * max_size, 10))
    height, width = 10, max_size * 10
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5
    res_with_seq = np.concatenate((cv2.resize(image_from_sequence('_ACM__', max_size=6), (10 * 7, 10)), result),
                                  axis=1)
    dict_sample_pred = {'real_class': y_sample, 'prediction': y_pred}

    return res_with_seq, dict_sample_pred, probs[0]


def get_kernel_activation_map(atcg_seq, new_X_seq, seq_id, all_filter_weights, kernel_id, max_size=300):
    kernel_size = all_filter_weights.shape[3]
    curr_filter_weights = all_filter_weights[kernel_id][0]
    opti_seq = get_letters(curr_filter_weights)

    res_filter_1 = apply_filter(new_X_seq[seq_id], curr_filter_weights)
    relu_res = np.array([max(res_filter_1[i], 0) for i in range(len(res_filter_1))])
    smooth_res = smooth_kernel_values(relu_res, kernel_size)
    # Create final image
    img = cv2.resize(image_from_sequence(atcg_seq[seq_id], max_size=max_size), (10 * max_size, 10))
    height, width = 10, max_size * 10
    smooth_res_img = (smooth_res - np.min(smooth_res)) / np.max(smooth_res)
    smooth_res_img = np.uint8(255 * smooth_res_img)
    heatmap = cv2.applyColorMap(cv2.resize(np.array([smooth_res_img]), (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    res_with_seq = np.concatenate(
        (cv2.resize(image_from_sequence(opti_seq + '_', max_size=kernel_size + 1), (10 * 7, 10)), result), axis=1)

    # cv2.imwrite('test.png', res_with_seq)

    return res_with_seq


# SUPPORT
def image_from_sequence(seq, max_size: int = 300):
    image_sequence = np.zeros((50, 0, 3))
    for index in range(0, max_size):
        if index < len(seq):
            bw_img = cv2.imread(
                "utils{}letters{}icons8-{}-50.png".format(slash, slash, seq[index].lower()),
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


def get_letter(arr):
    idx = np.argmax(arr)
    return 'ATCG'[idx]


def get_letters(arrs):
    letters = ''
    for column_id in range(arrs.shape[1]):
        letters = letters + get_letter(arrs[:, column_id])
    return letters


def apply_filter(variable_test, filter_weights):
    kernel_size = filter_weights.shape[1]
    conv_res = np.zeros(len(variable_test[0]))
    for i in range(len(variable_test[0]) - kernel_size):
        conv_res[i] = np.sum(variable_test[:, i:i + kernel_size] * filter_weights)
    return conv_res


def smooth_kernel_values(relu_res, kernel_size):
    smooth_res = np.zeros(len(relu_res))
    for i in range(len(relu_res) - kernel_size):
        values = relu_res[max(i - kernel_size + 1, 0):min(i + 1, len(relu_res))]
        smooth_res[i] = np.max(values)
    return smooth_res
