import math
from typing import List


def sigmoid(x: float) -> float:
    sigmoid_range = 34.538776394910684
    if x <= -1 * sigmoid_range:
        return 1e-15
    elif x >= sigmoid_range:
        return 1.0 - 1e-15
    else:
        return 1.0 / (1.0 + math.exp(-1 * x))


def softmax(x: List[float]) -> List[float]:
    n = len(x)
    max_x = max(x)
    numerator = [0.0] * n
    output = [0.0] * n
    denominator = 0.0

    for i in range(n):
        numerator[i] = math.exp(x[i] - max_x)
        denominator += numerator[i]

    for i in range(n):
        output[i] = numerator[i] / denominator

    return output


class LossFunc:
    """
    Base structure for loss functions.
    """

    def __init__(self):
        pass

    def get_loss(self, y_pred: List[List[float]], y: List[float]) -> float:
        raise NotImplementedError

    def get_grad(self, y_pred: List[List[float]], y: List[float]) -> List[List[float]]:
        raise NotImplementedError

    def get_hess(self, y_pred: List[List[float]], y: List[float]) -> List[List[float]]:
        raise NotImplementedError


class BCELoss(LossFunc):
    """
    Implementation of Binary Cross Entropy Loss.
    """

    def __init__(self):
        super().__init__()

    def get_loss(self, y_pred: List[List[float]], y: List[float]) -> float:
        loss = 0.0
        n = len(y_pred)
        for i in range(n):
            if y[i] == 1:
                loss += math.log(1 + math.exp(-1 * sigmoid(y_pred[i][0]))) / n
            else:
                loss += math.log(1 + math.exp(sigmoid(y_pred[i][0]))) / n
        return loss

    def get_grad(self, y_pred: List[List[float]], y: List[float]) -> List[List[float]]:
        element_num = len(y_pred)
        grad = [[0.0] for i in range(element_num)]
        for i in range(element_num):
            grad[i] = [sigmoid(y_pred[i][0]) - y[i]]
        return grad

    def get_hess(self, y_pred: List[List[float]], y: List[float]) -> List[List[float]]:
        element_num = len(y_pred)
        hess = [[0.0] for i in range(element_num)]
        for i in range(element_num):
            temp_proba = sigmoid(y_pred[i][0])
            hess[i] = [temp_proba * (1 - temp_proba)]
        return hess


class CELoss:
    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes

    def get_loss(self, y_pred: List[List[float]], y: List[float]) -> float:
        n = len(y_pred)
        y_pred_proba = [softmax(y_pred[i]) for i in range(n)]

        loss = 0
        for i in range(n):
            for c in range(self.num_classes):
                if y[i] == c:
                    loss -= math.log(y_pred_proba[i][c])
        return loss

    def get_grad(self, y_pred: List[List[float]], y: List[float]) -> List[List[float]]:
        n = len(y_pred)
        y_pred_proba = [softmax(y_pred[i]) for i in range(n)]

        grad = [[0] * self.num_classes for _ in range(n)]

        for i in range(n):
            for c in range(self.num_classes):
                grad[i][c] = y_pred_proba[i][c]
                if y[i] == c:
                    grad[i][c] -= 1
        return grad

    def get_hess(self, y_pred: List[List[float]], y: List[float]) -> List[List[float]]:
        n = len(y_pred)
        y_pred_proba = [softmax(y_pred[i]) for i in range(n)]

        hess = [[0] * self.num_classes for _ in range(n)]

        for i in range(n):
            for c in range(self.num_classes):
                hess[i][c] = y_pred_proba[i][c] * (1 - y_pred_proba[i][c])
        return hess
