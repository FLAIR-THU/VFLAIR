import time
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
import math
import random
import torch


def cosine_similarity(A, B):
    row_normalized_A = tf.math.l2_normalize(A, axis=1)
    row_normalized_B = tf.math.l2_normalize(B, axis=1)
    cosine_matrix = tf.linalg.matmul(
        row_normalized_A,
        row_normalized_B,
        adjoint_b=True)  # transpose second matrix)
    # average_abs_cosine = tf.reduce_mean(tf.math.abs(cosine_matrix), axis=1)
    # average_abs_cosine = tf.reduce_mean(cosine_matrix, axis=1)
    # print("cosine_matrix", cosine_matrix, type(cosine_matrix), cosine_matrix.shape)
    # assert 0==1
    return cosine_matrix


def update_acc(y, predicted_value):
    # print("update_acc", y[0:10], predicted_value[0:10], y[0].shape, predicted_value[0].shape)
    _y = y.numpy()
    _predicted_value = predicted_value.numpy()
    # print(f"[have nan, have inf] = [{torch.isnan(torch_predict_value).any()},{torch.isinf(torch_predict_value).any()}]")
    acc = (np.argmax(_predicted_value, axis=-1) == _y).sum() / len(_y)
    # print(acc)
    # assert 0==1
    # print("computed result", auc)
    # if auc:
    #     m_auc.update_state(auc)
    return acc


def update_auc(y, predicted_value, m_auc):
    # print("update_auc called")
    # print("y:", y)
    # print("predicted_value", predicted_value)
    torch_predict_value = torch.from_numpy(predicted_value.numpy())
    # print(f"[have nan, have inf] = [{torch.isnan(torch_predict_value).any()},{torch.isinf(torch_predict_value).any()}]")
    auc = compute_auc(y, predicted_value)
    # print("computed result", auc)
    # if auc:
    #     m_auc.update_state(auc)
    return auc


def compute_auc(y, predicted_value):
    # print("comput_auc called")
    # get rid of the 2nd dimension in  [n, 1]
    predicted_value = tf.reshape(predicted_value, shape=(-1))
    if tf.reduce_sum(y) == 0:  # no positive examples in this batch
        return None
    # m_auc.update_state(0.5) # currently set as 0.5 in some sense this is not well defined
    val_max = tf.math.reduce_max(predicted_value)
    val_min = tf.math.reduce_min(predicted_value)
    # print(f"[val_max, val_min] = [{val_max}, {val_min}]")
    pred = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)
    # create this is to avoid putting all different batches of examples in the same epoch together
    # auc_calculator = tf.keras.metrics.AUC()
    # auc_calculator.reset_states()
    # auc_calculator.update_state(y, pred)
    # auc = auc_calculator.result()
    auc = roc_auc_score(y_true=y.numpy(), y_score=pred.numpy(), multi_class='ovo')
    # print("keras.roc_auc_score is", auc)
    # if display and auc > 0.0:
    #     print('Alert')
    #     print(tf.reduce_max(predicted_value[y==1]))
    #     print(tf.reduce_min(predicted_value[y==0]))
    #     assert False
    return auc
