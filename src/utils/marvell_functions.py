import time
import sys, os

sys.path.append(os.pardir)

import torch
import tensorflow as tf
from . import constants as shared_var
import math
import random

# from utils import *

OBJECTIVE_EPSILON = 1e-16
CONVEX_EPSILON = 1e-20
NUM_CANDIDATE = 1


### for MARVELL ###
def symKL_objective(lam10, lam20, lam11, lam21, u, v, d, g):
    if (lam21 + v) == 0.0 or (lam20 + u) == 0.0 or (lam11 + v) == 0.0 or (lam10 + u) == 0.0:
        return float('inf')
    objective = (d - 1) * (lam20 + u) / (lam21 + v) \
                + (d - 1) * (lam21 + v) / (lam20 + u) \
                + (lam10 + u + g) / (lam11 + v) \
                + (lam11 + v + g) / (lam10 + u)
    return objective


def symKL_objective_zero_uv(lam10, lam11, g):
    objective = (lam10 + g) / lam11 \
                + (lam11 + g) / lam10
    return objective


def solve_isotropic_covariance(u, v, d, g, p, P,
                               lam10_init=None, lam20_init=None,
                               lam11_init=None, lam21_init=None):
    """ return the solution to the optimization problem
        Args:
        u ([type]): [the coordinate variance of the negative examples]
        v ([type]): [the coordinate variance of the positive examples]
        d ([type]): [the dimension of activation to protect]
        g ([type]): [squared 2-norm of g_0 - g_1, i.e. \|g^{(0)} - g^{(1)}\|_2^2]
        P ([type]): [the power constraint value]
    """

    if u == 0.0 and v == 0.0:
        return solve_zero_uv(g=g, p=p, P=P)

    ordering = [0, 1, 2]
    random.shuffle(x=ordering)

    solutions = []
    if u <= v:
        for i in range(NUM_CANDIDATE):
            if i % 3 == ordering[0]:
                # print('a')
                if lam20_init:  # if we pass an initialization
                    lam20 = lam20_init
                    # print('here')
                else:
                    lam20 = random.random() * P / (1 - p) / d
                lam10, lam11 = None, None
                # print('lam21', lam21)
            elif i % 3 == ordering[1]:
                # print('b')
                if lam11_init:
                    lam11 = lam11_init
                else:
                    lam11 = random.random() * P / p
                lam10, lam20 = None, None
                # print('lam11', lam11)
            else:
                # print('c')
                if lam10_init:
                    lam10 = lam10_init
                else:
                    lam10 = random.random() * P / (1 - p)
                lam11, lam20 = None, None
                # print('lam10', lam10)

            solutions.append(solve_small_neg(u=u, v=v, d=d, g=g, p=p, P=P, lam10=lam10, lam11=lam11, lam20=lam20))

    else:
        for i in range(NUM_CANDIDATE):
            if i % 3 == ordering[0]:
                if lam21_init:
                    lam21 = lam21_init
                else:
                    lam21 = random.random() * P / p / d
                lam10, lam11 = None, None
                # print('lam21', lam21)
            elif i % 3 == ordering[1]:
                if lam11_init:
                    lam11 = lam11_init
                else:
                    lam11 = random.random() * P / p
                lam10, lam21 = None, None
                # print('lam11', lam11)
            else:
                if lam10_init:
                    lam10 = lam10_init
                else:
                    lam10 = random.random() * P / (1 - p)
                lam11, lam21 = None, None
                # print('lam10', lam10)

            solutions.append(solve_small_pos(u=u, v=v, d=d, g=g, p=p, P=P, lam10=lam10, lam11=lam11, lam21=lam21))

    # print(solutions)
    lam10, lam20, lam11, lam21, objective = min(solutions, key=lambda x: x[-1])

    # print('sum', p * lam11 + p*(d-1)*lam21 + (1-p) * lam10 + (1-p)*(d-1)*lam20)

    return (lam10, lam20, lam11, lam21, objective)


def solve_zero_uv(g, p, P):
    C = P

    E = math.sqrt((C + (1 - p) * g) / (C + p * g))
    tau = max((P / p) / (E + (1 - p) / p), 0.0)
    # print('tau', tau)
    if 0 <= tau and tau <= P / (1 - p):
        # print('A')
        lam10 = tau
        lam11 = max(P / p - (1 - p) * tau / p, 0.0)
    else:
        # print('B')
        lam10_case1, lam11_case1 = 0.0, max(P / p, 0.0)
        lam10_case2, lam11_case2 = max(P / (1 - p), 0), 0.0
        objective1 = symKL_objective_zero_uv(lam10=lam10_case1, lam11=lam11_case1,
                                             g=g)
        objective2 = symKL_objective_zero_uv(lam10=lam10_case2, lam11=lam11_case2,
                                             g=g)
        if objective1 < objective2:
            lam10, lam11 = lam10_case1, lam11_case1
        else:
            lam10, lam11 = lam10_case2, lam11_case2

    objective = symKL_objective_zero_uv(lam10=lam10, lam11=lam11, g=g)
    # here we subtract d = 1 because the distribution is essentially one-dimensional
    return (lam10, 0.0, lam11, 0.0, 0.5 * objective - 1)


def solve_small_neg(u, v, d, g, p, P, lam10=None, lam20=None, lam11=None):
    """[When u < v]
    """
    # some intialization to start the alternating optimization
    LAM21 = 0.0
    i = 0
    objective_value_list = []

    if lam20:
        ordering = [0, 1, 2]
    elif lam11:
        ordering = [1, 0, 2]
    else:
        ordering = [1, 2, 0]
    # print(ordering)

    while True:
        if i % 3 == ordering[0]:  # fix lam20
            D = P - (1 - p) * (d - 1) * lam20
            C = D + p * v + (1 - p) * u

            E = math.sqrt((C + (1 - p) * g) / (C + p * g))
            tau = max((D / p + v - E * u) / (E + (1 - p) / p), 0.0)
            # print('tau', tau)
            if lam20 <= tau and tau <= P / (1 - p) - (d - 1) * lam20:
                # print('A')
                lam10 = tau
                lam11 = max(D / p - (1 - p) * tau / p, 0.0)
            else:
                # print('B')
                lam10_case1, lam11_case1 = lam20, max(P / p - (1 - p) * d * lam20 / p, 0.0)
                lam10_case2, lam11_case2 = max(P / (1 - p) - (d - 1) * lam20, 0), 0.0
                objective1 = symKL_objective(lam10=lam10_case1, lam20=lam20, lam11=lam11_case1, lam21=LAM21,
                                             u=u, v=v, d=d, g=g)
                objective2 = symKL_objective(lam10=lam10_case2, lam20=lam20, lam11=lam11_case2, lam21=LAM21,
                                             u=u, v=v, d=d, g=g)
                if objective1 < objective2:
                    lam10, lam11 = lam10_case1, lam11_case1
                else:
                    lam10, lam11 = lam10_case2, lam11_case2

        elif i % 3 == ordering[1]:  # fix lam11
            D = max((P - p * lam11) / (1 - p), 0.0)
            f = lambda x: symKL_objective(lam10=D - (d - 1) * x, lam20=x, lam11=lam11, lam21=LAM21,
                                          u=u, v=v, d=d, g=g)

            def f_prime(x):
                if x == 0.0 and u == 0.0:
                    return float('-inf')
                else:
                    return (d - 1) / v - (d - 1) / (lam11 + v) - (d - 1) / (x + u) * (v / (x + u)) + (lam11 + v + g) / (
                                D - (d - 1) * x + u) * ((d - 1) / (D - (d - 1) * x + u))

            # print('D/d', D/d)
            lam20 = convex_min_1d(xl=0.0, xr=D / d, f=f, f_prime=f_prime)
            lam10 = max(D - (d - 1) * lam20, 0.0)

        else:  # fix lam10
            D = max(P - (1 - p) * lam10, 0.0)  # avoid negative due to numerical error
            f = lambda x: symKL_objective(lam10=lam10, lam20=x, lam11=D / p - (1 - p) * (d - 1) * x / p, lam21=LAM21,
                                          u=u, v=v, d=d, g=g)

            def f_prime(x):
                if x == 0.0 and u == 0.0:
                    return float('-inf')
                else:
                    return (d - 1) / v - (1 - p) * (d - 1) / (lam10 + u) / p - (d - 1) / (x + u) * (v / (x + u)) + (
                                lam10 + u + g) / (D / p - (1 - p) * (d - 1) * x / p + v) * (1 - p) * (d - 1) / p / (
                                D / p - (1 - p) * (d - 1) * x / p + v)

            # print('lam10', 'D/((1-p)*(d-1)', lam10, D/((1-p)*(d-1)))
            lam20 = convex_min_1d(xl=0.0, xr=min(D / ((1 - p) * (d - 1)), lam10), f=f, f_prime=f_prime)
            lam11 = max(D / p - (1 - p) * (d - 1) * lam20 / p, 0.0)

        if lam10 < 0 or lam20 < 0 or lam11 < 0 or LAM21 < 0:  # check to make sure no negative values
            assert False, i

        objective_value_list.append(symKL_objective(lam10=lam10, lam20=lam20, lam11=lam11, lam21=LAM21,
                                                    u=u, v=v, d=d, g=g))
        if (i >= 3 and objective_value_list[-4] - objective_value_list[-1] < OBJECTIVE_EPSILON) or i >= 100:
            # print(i)
            return lam10, lam20, lam11, LAM21, 0.5 * objective_value_list[-1] - d

        i += 1


def solve_small_pos(u, v, d, g, p, P, lam10=None, lam11=None, lam21=None):
    """[When u > v] lam20 = 0.0 and will not change throughout the optimization
    """
    # some intialization to start the alternating optimization
    LAM20 = 0.0
    i = 0
    objective_value_list = []
    if lam21:
        ordering = [0, 1, 2]
    elif lam11:
        ordering = [1, 0, 2]
    else:
        ordering = [1, 2, 0]
    # print(ordering)
    while True:
        if i % 3 == ordering[0]:  # fix lam21
            D = P - p * (d - 1) * lam21
            C = D + p * v + (1 - p) * u

            E = math.sqrt((C + (1 - p) * g) / (C + p * g))
            tau = max((D / p + v - E * u) / (E + (1 - p) / p), 0.0)
            # print('tau', tau)
            if 0.0 <= tau and tau <= (P - p * d * lam21) / (1 - p):
                # print('A')
                lam10 = tau
                lam11 = max(D / p - (1 - p) * tau / p, 0.0)
            else:
                # print('B')
                lam10_case1, lam11_case1 = 0, max(P / p - (d - 1) * lam21, 0.0)
                lam10_case2, lam11_case2 = max((P - p * d * lam21) / (1 - p), 0.0), lam21
                objective1 = symKL_objective(lam10=lam10_case1, lam20=LAM20, lam11=lam11_case1, lam21=lam21,
                                             u=u, v=v, d=d, g=g)
                objective2 = symKL_objective(lam10=lam10_case2, lam20=LAM20, lam11=lam11_case2, lam21=lam21,
                                             u=u, v=v, d=d, g=g)
                if objective1 < objective2:
                    lam10, lam11 = lam10_case1, lam11_case1
                else:
                    lam10, lam11 = lam10_case2, lam11_case2

        elif i % 3 == ordering[1]:  # fix lam11
            D = max(P - p * lam11, 0.0)
            f = lambda x: symKL_objective(lam10=(D - p * (d - 1) * x) / (1 - p), lam20=LAM20, lam11=lam11, lam21=x,
                                          u=u, v=v, d=d, g=g)

            def f_prime(x):
                if x == 0.0 and v == 0.0:
                    return float('-inf')
                else:
                    return (d - 1) / u - p * (d - 1) / (lam11 + v) / (1 - p) - (d - 1) / (x + v) * (u / (x + v)) + (
                                lam11 + v + g) / ((D - p * (d - 1) * x) / (1 - p) + u) * p * (d - 1) / (1 - p) / (
                                (D - p * (d - 1) * x) / (1 - p) + u)

            # print('lam11', 'D/p/(d-1)', lam11, D/p/(d-1))
            lam21 = convex_min_1d(xl=0.0, xr=min(D / p / (d - 1), lam11), f=f, f_prime=f_prime)
            lam10 = max((D - p * (d - 1) * lam21) / (1 - p), 0.0)

        else:  # fix lam10
            D = max((P - (1 - p) * lam10) / p, 0.0)
            f = lambda x: symKL_objective(lam10=lam10, lam20=LAM20, lam11=D - (d - 1) * x, lam21=x,
                                          u=u, v=v, d=d, g=g)

            def f_prime(x):
                if x == 0.0 and v == 0.0:
                    return float('-inf')
                else:
                    return (d - 1) / u - (d - 1) / (lam10 + u) - (d - 1) / (x + v) * (u / (x + v)) + (lam10 + u + g) / (
                                D - (d - 1) * x + v) * (d - 1) / (D - (d - 1) * x + v)

            lam21 = convex_min_1d(xl=0.0, xr=D / d, f=f, f_prime=f_prime)
            lam11 = max(D - (d - 1) * lam21, 0.0)

        if lam10 < 0 or LAM20 < 0 or lam11 < 0 or lam21 < 0:
            assert False, i

        objective_value_list.append(symKL_objective(lam10=lam10, lam20=LAM20, lam11=lam11, lam21=lam21,
                                                    u=u, v=v, d=d, g=g))

        if (i >= 3 and objective_value_list[-4] - objective_value_list[-1] < OBJECTIVE_EPSILON) or i >= 100:
            # print(i)
            return lam10, LAM20, lam11, lam21, 0.5 * objective_value_list[-1] - d

        i += 1


def convex_min_1d(xl, xr, f, f_prime):
    # print('xl, xr', xl, xr)
    assert xr <= 1e5
    assert xl <= xr, (xl, xr)
    # print('xl, xr', xl, xr)

    xm = (xl + xr) / 2
    # print('xl', xl, f(xl), f_prime(xl))
    # print('xr', xr, f(xr), f_prime(xr))
    if abs(xl - xr) <= CONVEX_EPSILON:
        return min((f(x), x) for x in [xl, xm, xr])[1]
    if f_prime(xl) <= 0 and f_prime(xr) <= 0:
        return xr
    elif f_prime(xl) >= 0 and f_prime(xr) >= 0:
        return xl
    if f_prime(xm) > 0:
        return convex_min_1d(xl=xl, xr=xm, f=f, f_prime=f_prime)
    else:
        return convex_min_1d(xl=xm, xr=xr, f=f, f_prime=f_prime)


def small_neg_problem_string(u, v, d, g, p, P):
    return 'minimize ({2}-1)*(z + {0})/{1} + ({2}-1)*{1}/(z+{0})+(x+{0}+{3})/(y+{1}) + (y+{1}+{3})/(x+{0}) subject to x>=0, y>=0, z>=0, z<=x, {4}*y+(1-{4})*x+(1-{4})*({2}-1)*z={5}'.format(
        u, v, d, g, p, P)


def small_pos_problem_string(u, v, d, g, p, P):
    return 'minimize ({2}-1)*{0}/(z+{1}) + ({2}-1)*(z + {1})/{0} + (x+{0}+{3})/(y+{1}) + (y+{1}+{3})/(x+{0}) subject to x>=0, y>=0, z>=0, z<=y, {4}*y+(1-{4})*x+{4}*({2}-1)*z={5}'.format(
        u, v, d, g, p, P)


def zero_uv_problem_string(g, p, P):
    return 'minimize (x+{0})/y + (y+{0})/x subject to x>=0, y>=0, {1}*y+(1-{1})*x={2}'.format(g, p, P)


# if __name__ == '__main__':
#     import random
#     import time
#     from collections import Counter

#     test_neg = False

#     # u=random.random()
#     # v=random.uniform(u, u+random.random())
#     # u = 0.0
#     # v = 0.0
#     # d=random.randint(2, 1000)
#     # g=random.random() * d
#     # p=random.random() * 0.5
#     # p=0.25

#     # P=random.random() * d

#     # r = lambda x: round(x, 2)


#     # u,v,g,p,P = r(u), r(v), r(g), r(p), r(P)
#     # u,v,g,p,P = r(v), r(u), r(g), r(p), r(P) # make sure u > v
#     u = 3.229033590534426e-15
#     v = 3.0662190349955726e-15
#     d = 128.0
#     g = 5.015613264502392e-10
#     p = 0.253936767578125
#     P = 2328365.0213796967

#     print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g,p,P))
#     start = time.time()
#     lam10, lam20, lam11, lam21, sumKL = solve_isotropic_covariance(u=u, v=v, d=d, g=g, p=p, P=P)
#     print(lam10, lam20, lam11, lam21, sumKL)
#     print('time', time.time() - start)
#     if u < v:
#         print(small_neg_problem_string(u=u,v=v,d=d,g=g,p=p,P=P))
#     else:
#         print(small_pos_problem_string(u=u,v=v,d=d,g=g,p=p,P=P))

#     start = time.time()
#     print(solve_isotropic_covariance(u=u, v=v, d=d, g=g, p=p, P=P + 10, 
#                                      lam10_init=lam10, lam20_init=lam20,
#                                      lam11_init=lam11, lam21_init=lam21))
#     print('time', time.time() - start)


def KL_gradient_perturb(g, classes, sumKL_threshold, dynamic=False, init_scale=1.0, uv_choice='uv', p_frac='pos_frac'):
    assert len(classes) == 2

    # sess = tf.compat.v1.Session()

    # the batch label was stored in shared_var.batch_y in train_and_test
    # print('start')
    # start = time.time()

    # g is a torch.tensor
    # Torch => numpy
    numpy_g = g.cpu().numpy()
    # numpy => Tensorflow
    g = tf.convert_to_tensor(numpy_g)

    g_original_shape = g.shape
    g = tf.reshape(g, shape=(g_original_shape[0], -1))
    # print(g)

    _y = shared_var.batch_y
    # y = y.cpu().numpy()
    y = _y
    # y = tf.as_dtype(_y)
    pos_g = g[y == 1]
    pos_g_mean = tf.math.reduce_mean(pos_g, axis=0, keepdims=True)  # shape [1, d]
    pos_coordinate_var = tf.reduce_mean(tf.math.square(pos_g - pos_g_mean), axis=0)  # use broadcast
    neg_g = g[y == 0]
    neg_g_mean = tf.math.reduce_mean(neg_g, axis=0, keepdims=True)  # shape [1, d]
    neg_coordinate_var = tf.reduce_mean(tf.math.square(neg_g - neg_g_mean), axis=0)
    # print("pos_g:",pos_g, pos_g_mean, pos_coordinate_var)
    # print("neg_g:",neg_g, neg_g_mean, neg_coordinate_var)
    # print("pos_g:",pos_g)
    # print("neg_g:",neg_g)

    avg_pos_coordinate_var = tf.reduce_mean(pos_coordinate_var)
    avg_neg_coordinate_var = tf.reduce_mean(neg_coordinate_var)
    # print('pos', avg_pos_coordinate_var)
    # print('neg', avg_neg_coordinate_var)

    g_diff = pos_g_mean - neg_g_mean
    g_diff_norm = float(tf.norm(tensor=g_diff).numpy())
    # if g_diff_norm ** 2 > 1:
    #     print('pos_g_mean', pos_g_mean.shape)
    #     print('neg_g_mean', neg_g_mean.shape)
    #     assert g_diff_norm

    if uv_choice == 'uv':
        u = float(avg_neg_coordinate_var)
        v = float(avg_pos_coordinate_var)
        if u == 0.0:
            print('neg_g')
            print(neg_g)
        if v == 0.0:
            print('pos_g')
            print(pos_g)
    elif uv_choice == 'same':
        u = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
        v = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
    elif uv_choice == 'zero':
        u, v = 0.0, 0.0

    d = float(g.shape[1])

    if p_frac == 'pos_frac':
        p = float(tf.reduce_sum(y) / len(y))  # p is set as the fraction of positive in the batch
    else:
        p = float(p_frac)

    scale = init_scale
    P = scale * g_diff_norm ** 2
    # print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g_diff_norm**2,p,P))

    # print('compute problem instance', time.time() - start)
    # start = time.time()

    lam10, lam20, lam11, lam21 = None, None, None, None
    while True:
        P = scale * g_diff_norm ** 2
        # print('g_diff_norm ** 2', g_diff_norm ** 2)
        # print('P', P)
        # print('u, v, d, p', u, v, d, p)
        lam10, lam20, lam11, lam21, sumKL = \
            solve_isotropic_covariance(
                u=u,
                v=v,
                d=d,
                g=g_diff_norm ** 2,
                p=p,
                P=P,
                lam10_init=lam10,
                lam20_init=lam20,
                lam11_init=lam11,
                lam21_init=lam21)
        # print('sumKL', sumKL)
        # print()

        # print(scale)
        if not dynamic or sumKL <= sumKL_threshold:
            break

        scale *= 1.5  # loosen the power constraint

    # print('solving time', time.time() - start)
    # start = time.time()

    with shared_var.writer.as_default():
        tf.summary.scalar(name='solver/u',
                          data=u,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/v',
                          data=v,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/g',
                          data=g_diff_norm ** 2,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/p',
                          data=p,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/scale',
                          data=scale,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/P',
                          data=P,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/lam10',
                          data=lam10,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/lam20',
                          data=lam20,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/lam11',
                          data=lam11,
                          step=shared_var.counter)
        tf.summary.scalar(name='solver/lam21',
                          data=lam21,
                          step=shared_var.counter)
        # tf.summary.scalar(name='sumKL_before',
        #                 data=symKL_objective(lam10=0.0,lam20=0.0,lam11=0.0,lam21=0.0,
        #                                     u=u, v=v, d=d, g=g_diff_norm**2),
        #                 step=shared_var.counter)
        # even if we didn't use avg_neg_coordinate_var for u and avg_pos_coordinate_var for v, we use it to evaluate the sumKL_before
        tf.summary.scalar(name='sumKL_before',
                          data=symKL_objective(lam10=0.0, lam20=0.0, lam11=0.0, lam21=0.0,
                                               u=float(avg_neg_coordinate_var),
                                               v=float(avg_pos_coordinate_var),
                                               d=d, g=g_diff_norm ** 2),
                          step=shared_var.counter)
        tf.summary.scalar(name='sumKL_after',
                          data=sumKL,
                          step=shared_var.counter)
        tf.summary.scalar(name='error prob lower bound',
                          data=0.5 - math.sqrt(sumKL) / 4,
                          step=shared_var.counter)

    # print('tb logging', time.time() - start)
    # start = time.time()

    perturbed_g = g
    y_float = tf.cast(y, dtype=tf.float32)

    # positive examples add noise in g1 - g0
    perturbed_g += tf.reshape(tf.multiply(x=tf.random.normal(shape=y.shape),
                                          y=y_float), shape=(-1, 1)) * g_diff * (math.sqrt(lam11 - lam21) / g_diff_norm)

    # add spherical noise to positive examples
    if lam21 > 0.0:
        perturbed_g += tf.random.normal(shape=g.shape) * tf.reshape(y_float, shape=(-1, 1)) * math.sqrt(lam21)

    # negative examples add noise in g1 - g0
    perturbed_g += tf.reshape(tf.multiply(x=tf.random.normal(shape=y.shape),
                                          y=1 - y_float), shape=(-1, 1)) * g_diff * (
                               math.sqrt(lam10 - lam20) / g_diff_norm)

    # add spherical noise to negative examples
    if lam20 > 0.0:
        perturbed_g += tf.random.normal(shape=g.shape) * tf.reshape(1 - y_float, shape=(-1, 1)) * math.sqrt(lam20)

    # print('noise adding', time.time() - start)

    # print('a')
    # print(perturbed_g)
    # print('b')
    # print(perturbed_g[y==1])
    # print('c')
    # print(perturbed_g[y==0])

    '''
    pos_cov = tf.linalg.matmul(a=g[y==1] - pos_g_mean, b=g[y==1] - pos_g_mean, transpose_a=True) / g[y==1].shape[0]
    print('pos_var', pos_coordinate_var)
    print('pos_cov', pos_cov)
    print('raw svd', tf.linalg.svd(pos_cov, compute_uv=False))
    print('diff svd', tf.linalg.svd(pos_cov - tf.linalg.tensor_diag(pos_coordinate_var), compute_uv=False))
    # assert False
    '''
    # if shared_var.counter < 2000:
    #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_unperturbed',
    #             arr=g.numpy())
    #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_perturbed',
    #             arr=perturbed_g.numpy())
    #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_label',
    #             arr=shared_var.batch_y.numpy())

    tf_tensor_result = tf.reshape(perturbed_g, shape=g_original_shape)
    # Tensorflow => Numpy
    numpy_result = tf_tensor_result.numpy()
    # Numpy => Torch
    torch_result = torch.from_numpy(numpy_result)
    return torch_result
