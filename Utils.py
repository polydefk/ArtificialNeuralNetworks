import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, zero_one_loss, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.random.seed(0)


def create_non_linearly_separable_data(n=100, use_validation_set=False, percent_split=0.2):
    meanA = [1, 0.5]
    meanB = [-0.5, -1]

    sigmaA = np.diag([1, 1])
    sigmaB = np.diag([1, 1])

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    labelsA = -np.ones((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, n)
    labelsB = np.ones((classB.shape[0], 1))

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    [inputs, inputs_labels] = shuffle(X, Y)

    if use_validation_set:
        [inputs, input_validation, inputs_labels, input_validation_labels] = \
            train_test_split(X, Y, test_size=percent_split, random_state=42)

        return [inputs.T, inputs_labels, input_validation.T, input_validation_labels]

    return [inputs.T, inputs_labels, None, None]


def create_one_out_of_n_dataset(n=8):
    data = -np.ones((n, n))
    np.fill_diagonal(data, 1)
    # labels = np.zeros((n,n))
    # np.fill_diagonal(labels, 1)

    return [data, data]


def plot_initial_data(inputs, targets):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=15)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=15)

    plt.ylim(-10, 10)
    plt.xlim(-6, 6)

    plt.show()


def plot_error(error, legend_names, num_epochs):
    # fig config
    plt.figure()
    plt.grid(True)

    plt.ylim(0, 1)
    plt.xlim(-0.5, num_epochs)

    epochs = np.arange(0, num_epochs, 1)

    for i in range(len(error)):
        plt.plot(epochs, error[i][:])

    plt.legend(legend_names, loc='upper right')

    plt.show()


def compute_error(targets, predictions, binary):
    mse = mean_squared_error(targets, predictions)
    loss = 0
    # fraction of misclassifications
    if binary:
        predictions = np.where(predictions >= 0, 1, -1)
        loss = zero_one_loss(targets, predictions, normalize=True)

    return loss * 100, mse


def plot_3d_data(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()