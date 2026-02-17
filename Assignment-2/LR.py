import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## Data loader and data generation functions
def data_loader(args):
    """
    Output:
        X: the data matrix (numpy array) of size 1-by-N
        Y: the label matrix (numpy array) of size N-by-1
    """
    if args.data == "linear":
        print("Using linear")
        X, Y = data_linear()
    elif args.data == "quadratic":
        print("Using quadratic")
        X, Y = data_quadratic()
    else:
        print("Using simple")
        X, Y = data_simple()
    return X, Y


def data_linear():
    data = np.load(osp.join(args.path, 'Linear.npz'))
    X = data['X']
    Y = data['Y']
    return X, Y


def data_quadratic():
    data = np.load(osp.join(args.path, 'Quadratic.npz'))
    X = data['X']
    Y = data['Y']
    return X, Y


def data_simple():
    N = 20
    X = np.linspace(0.0, 10.0, num=N).reshape(1, N)
    Y = np.linspace(1.0, 3.0, num=N).reshape(N, 1)
    return X, Y


def display_LR(args, w, b, Y, Y_test, X_original):
    N = X_original.shape[1]
    Y = np.concatenate((Y, Y_test), 0)
    phi = np.ones(N)
    phi[:int(0.7*N)] = 0
    x_min = np.min(X_original)
    x_max = np.max(X_original)
    x = np.linspace(x_min-0.05, x_max+0.05, num=1000)
    XX = polynomial_transform(x.reshape(1,-1), args.polynomial)
    YY = np.matmul(w.transpose(), XX) + b
    plt.scatter(X_original.reshape(-1), Y.reshape(-1), c=phi, cmap=plt.cm.Spectral)
    plt.plot(x.reshape(-1), YY.reshape(-1), color='black', linewidth=3)
    if args.save:
        plt.savefig(args.data + '_' + str(args.polynomial) + '.png', format='png')
        np.savez('Results_' + args.data + '_' + str(args.polynomial) + '.npz', w=w, b=b)
    plt.show()
    plt.close()


def auto_grade(w, b):
    print("In auto grader!")
    if w.ndim != 2:
        print("Wrong dimensionality of w")
    else:
        if w.shape[0] != 2 or w.shape[1] != 1:
            print("Wrong shape of w")
        else:
            if sum((w - [[2.00000000e-01], [2.77555756e-17]]) ** 2) < 10 ** -6:
                print("Correct w")
            else:
                print("Incorrect w")

    if (b - 1) ** 2 < 10 ** -6:
        print("Correct b")
    else:
        print("Incorrect b")


def linear_regression(X, Y):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label
    Output:
        w: the linear weight vector. Please represent it as a D-by-1 matrix (numpy array).
        b: the bias (a scalar)
    """
    X = np.copy(X)
    D = X.shape[0]
    N = X.shape[1]
    tilde_X = np.concatenate((X, np.ones((1, N))), 0)

    tilde_w = np.matmul(np.linalg.inv(np.matmul(tilde_X, tilde_X.transpose())),
                        np.matmul(tilde_X, Y))
    w = tilde_w[:D]
    b = tilde_w[D]
    return w, b


def polynomial_transform(X, degree_polynomial):
    """
    Input:
        X: a 1-by-N matrix (numpy array) of the input data
        degree_polynomial: polynomial degree
    Output:
        X_out: a degree_polynomial-by-N matrix (numpy array)
    """
    X = np.copy(X)
    N = X.shape[1]
    X_out = np.zeros((degree_polynomial, N))
    for d in range(degree_polynomial):
        X_out[d, :] = X.reshape(-1) ** (d + 1)
    return X_out


def main(args):
    if args.auto_grade:
        args.data = "simple"
        args.polynomial = int(2)
        args.display = False
        args.save = False

    X_original, Y = data_loader(args)
    X = polynomial_transform(np.copy(X_original), args.polynomial)

    N = X.shape[1]
    X_test = X[:, int(0.7*N):]
    X = X[:, :int(0.7*N)]
    Y_test = Y[int(0.7 * N):, :]
    Y = Y[:int(0.7 * N), :]

    w, b = linear_regression(X, Y)
    print("w: ", w)
    print("b: ", b)

    training_error = np.mean((np.matmul(w.transpose(), X) + b - Y.transpose()) ** 2)
    test_error = np.mean((np.matmul(w.transpose(), X_test) + b - Y_test.transpose()) ** 2)
    print("Training mean square error: ", training_error)
    print("Test mean square error: ", test_error)

    if args.display:
        display_LR(args, w, b, Y, Y_test, X_original)

    if args.auto_grade:
        auto_grade(w, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running linear regression (LR)")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="linear", type=str)
    parser.add_argument('--polynomial', default=1, type=int)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--auto_grade', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
