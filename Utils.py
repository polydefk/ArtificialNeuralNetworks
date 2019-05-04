import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, zero_one_loss
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.random.seed(0)


def create_linearly_separable_data():
    meanA = [2, 1.5]
    meanB = [-2.5, -2.5]

    sigmaA = [[1, 0], [0, 0.5]]
    sigmaB = [[-1, 0], [0, -1]]

    n = 100

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    labelsA = -np.ones((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, n)
    labelsB = np.ones((classB.shape[0], 1))

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    return [X, Y]


def create_non_linearly_separable_data_2(ndata=100, use_validation_set=False, case=1):
    mA = [1.0, 0.3]
    mB = [0.0, -0.1]

    sigmaA = 0.2
    sigmaB = 0.3

    classA = np.hstack(
        [np.random.randn(1, int(ndata / 2)) * sigmaA - mA[0], np.random.randn(1, int(ndata / 2)) * sigmaA + mA[0]])
    classA = np.vstack([classA, np.random.randn(1, ndata) * sigmaA + mA[1]])

    classB = np.random.randn(1, ndata) * sigmaB + mB[0]
    classB = np.vstack([classB, np.random.randn(1, ndata) * sigmaB + mB[1]])

    labelsA = -np.ones((np.shape(classA)[1], 1))
    labelsB = np.ones((np.shape(classB)[1], 1))

    classA = classA.T
    classB = classB.T

    if use_validation_set:

        if case == 1:
            [inputsA, input_validationA, inputs_labelsA, input_validation_labelsA] = \
                remove_percent_of_data(classA, labelsA, percent=0.25)

            [inputsB, input_validationB, inputs_labelsB, input_validation_labelsB] = \
                remove_percent_of_data(classB, labelsB, percent=0.25)

            inputs = np.concatenate((inputsA, inputsB), axis=0)
            inputs_labels = np.concatenate((inputs_labelsA, inputs_labelsB))

            input_validation = np.concatenate((input_validationA, input_validationB), axis=0)
            input_validation_labels = np.concatenate((input_validation_labelsA, input_validation_labelsB))

        elif case == 2:
            [inputsA, input_validationA, inputs_labelsA, input_validation_labelsA] = \
                remove_percent_of_data(classA, labelsA, percent=0.5)

            inputs = np.concatenate((inputsA, classB), axis=0)
            inputs_labels = np.concatenate((inputs_labelsA, labelsB))

            input_validation = input_validationA
            input_validation_labels = input_validation_labelsA

        elif case == 3:
            [inputsB, input_validationB, inputs_labelsB, input_validation_labelsB] = \
                remove_percent_of_data(classB, labelsB, percent=0.5)

            inputs = np.concatenate((classA, inputsB), axis=0)
            inputs_labels = np.concatenate((labelsA, inputs_labelsB))

            input_validation = input_validationB
            input_validation_labels = input_validation_labelsB
        elif case == 4:

            [inputsA, input_validationA, inputs_labelsA, input_validation_labelsA] = \
                reduce_20_80(classA, labelsA)

            inputs = np.concatenate((inputsA, classB), axis=0)
            inputs_labels = np.concatenate((inputs_labelsA, labelsB))

            input_validation = input_validationA
            input_validation_labels = input_validation_labelsA
        else:
            raise Exception("cannot find given case")

        return [inputs.T, inputs_labels, input_validation.T, input_validation_labels]

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    [inputs, inputs_labels] = shuffle(X, Y)

    return [inputs.T, inputs_labels, None, None]


def remove_percent_of_data(dataset, labels, percent=0.25):
    keep_data = 1 - percent

    [dataset, labels] = shuffle(dataset, labels)
    inputs = dataset[0:round(len(dataset) * keep_data)]
    inputs_labels = labels[0:round(len(dataset) * keep_data)]

    input_validation = dataset[round(len(dataset) * keep_data):]
    input_validation_labels = labels[round(len(dataset) * keep_data):]

    return [inputs, input_validation, inputs_labels, input_validation_labels]


def reduce_20_80(dataset, labelset):
    dataset, labelset = shuffle(dataset, labelset)

    ind_train = np.argwhere(dataset[:, 0] > 0)
    ind_test = np.argwhere(dataset[:, 0] < 0)

    ind_train = ind_train[0:round(len(ind_train) * 0.8)]
    ind_test = ind_test[0:round(len(ind_test) * 0.2)]

    inputs = dataset[ind_train]

    input_validation = dataset[ind_test]

    input_validation_labels = labelset[ind_test]
    inputs_labels = labelset[ind_train]

    inputs = np.reshape(inputs, (len(inputs), 2))
    input_validation = np.reshape(input_validation, (len(input_validation), 2))
    input_validation_labels = np.reshape(input_validation_labels, (len(input_validation_labels), 1))
    inputs_labels = np.reshape(inputs_labels, (len(inputs_labels), 1))

    return [inputs, input_validation, inputs_labels, input_validation_labels]


# case 1 is random 25% from each class
# case 2 random 50% from classA
# case 3 random 50% from classB
# case 4 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0
def create_non_linearly_separable_data(n=100, use_validation_set=False, case=1):
    meanA = [1, 0.5]
    meanB = [-0.5, -1]

    sigmaA = np.diag([1, 1])
    sigmaB = np.diag([1, 1])

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    labelsA = -np.ones((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, n)
    labelsB = np.ones((classB.shape[0], 1))

    if use_validation_set:
        if case == 1:
            [inputsA, input_validationA, inputs_labelsA, input_validation_labelsA] = \
                remove_percent_of_data(classA, labelsA, percent=0.25)

            [inputsB, input_validationB, inputs_labelsB, input_validation_labelsB] = \
                remove_percent_of_data(classB, labelsB, percent=0.25)

            inputs = np.concatenate((inputsA, inputsB), axis=0)
            inputs_labels = np.concatenate((inputs_labelsA, inputs_labelsB))

            input_validation = np.concatenate((input_validationA, input_validationB), axis=0)
            input_validation_labels = np.concatenate((input_validation_labelsA, input_validation_labelsB))

        elif case == 2:
            [inputsA, input_validationA, inputs_labelsA, input_validation_labelsA] = \
                remove_percent_of_data(classA, labelsA, percent=0.5)

            inputs = np.concatenate((inputsA, classB), axis=0)
            inputs_labels = np.concatenate((inputs_labelsA, labelsB))

            input_validation = input_validationA
            input_validation_labels = input_validation_labelsA

        elif case == 3:
            [inputsB, input_validationB, inputs_labelsB, input_validation_labelsB] = \
                remove_percent_of_data(classB, labelsB, percent=0.5)

            inputs = np.concatenate((classA, inputsB), axis=0)
            inputs_labels = np.concatenate((labelsA, inputs_labelsB))

            input_validation = input_validationB
            input_validation_labels = input_validation_labelsB
        elif case == 4:

            [inputsA, input_validationA, inputs_labelsA, input_validation_labelsA] = \
                reduce_20_80(classA, labelsA)

            inputs = np.concatenate((inputsA, classB), axis=0)
            inputs_labels = np.concatenate((inputs_labelsA, labelsB))

            input_validation = input_validationA
            input_validation_labels = input_validation_labelsA
        else:
            raise Exception("cannot find given case")

        return [inputs.T, inputs_labels, input_validation.T, input_validation_labels]

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    [inputs, inputs_labels] = shuffle(X, Y)

    return [inputs.T, inputs_labels, None, None]


def create_one_out_of_n_dataset(n=8):
    data = -np.ones((n, n))
    np.fill_diagonal(data, 1)
    # labels = np.zeros((n,n))
    # np.fill_diagonal(labels, 1)

    return [data, data]


def compute_error(targets, predictions, binary):
    mse = mean_squared_error(targets, predictions)
    loss = 0
    # fraction of misclassifications
    if binary:
        predictions = np.where(predictions >= 0, 1, -1)
        loss = zero_one_loss(targets, predictions, normalize=True)

    return loss, mse


def plot_initial_data(inputs, targets):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=15)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=15)

    # plt.ylim(-10, 10)
    # plt.xlim(-6, 6)

    plt.show()

def plot_nn_with_nodes(error, legend_names, num_epochs, title):
    # fig config
    plt.figure()
    plt.grid(True)

    plt.ylim(0.01, 0.07)
    plt.xlim(0, len(num_epochs))

    # epochs = np.arange(0, num_epochs, 1)

    for i in range(len(error)):
        plt.plot(num_epochs, error[i][:])

    plt.xlabel('Number of nodes')
    plt.ylabel('Error (mse)')

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()

def plot_weights_distribution(regularazations, weights_layer_1):


    for i in range(len(regularazations)):
        plt.hist(weights_layer_1[i], 30)
        plt.title('Weight distribution with lr = {0}'.format(str(regularazations[i])))
        plt.xlim(-1.1, 1.1)
        plt.show()
        print(np.sum(weights_layer_1[i] ** 2))


def plot_error_with_epochs(error, legend_names, num_epochs, title):
    # fig config
    plt.figure()
    plt.grid(True)

    # plt.ylim(0, 4)
    plt.xlim(-0.5, num_epochs)

    epochs = np.arange(0, len(error[0]), 1)

    for i in range(len(error)):
        plt.plot(epochs, error[i][:])

    plt.xlabel('Epochs')
    plt.ylabel('Error (mse)')

    plt.title(title)
    plt.show()


def plot_learning_curves(error, legend_names, train_size, title, loss):
    # fig config
    plt.figure()
    plt.grid(True)


    for i in range(len(error)):
        plt.plot(train_size, error[i][:])

    for i in range(len(error)):
        plt.plot(train_size, loss[i][:])


    plt.xlabel('Training percentage')
    plt.ylabel('Error')

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()


def plot_error_hidden_nodes(error, legend_names, hidden_nodes, title, loss):
    # fig config
    plt.figure()
    plt.grid(True)

    # plt.ylim(0, 2)
    # plt.xlim(-0.5, num_epochs)

    for i in range(len(error)):
        plt.plot(hidden_nodes, error[i][:])

    for i in range(len(error)):
        plt.plot(hidden_nodes, loss[i][:])

    # plt.plot(hidden_nodes, error)
    # plt.plot(hidden_nodes, loss)

    plt.xlabel('Nodes in hidden layer')
    plt.ylabel('Error')

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()


def plot_3d_data(X, Y, Z, pause=True):
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

    if pause:
        plt.pause(interval=.1)
    else:
        plt.show()


def plot_glass_data_prediction(pred, y_test, title):
    # fig config
    fig = plt.figure()
    plt.grid(True)
    epochs = np.arange(0, len(pred), 1)

    plt.title(title)
    plt.plot(epochs, pred, color='r', label='Prediction')
    plt.plot(epochs, y_test, color='b', label='Actual data')
    plt.legend()
    plt.show()


def plot_Perceptron(inputs, targets, weights, title):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=10)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=10)

    plt.title(title)

    plt.ylim(-10, 10)
    plt.xlim(-6, 6)

    xx = np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1]))
    slope = -(weights[2] / weights[1]) / (weights[2] / weights[0])
    intercept = -weights[2] / weights[1]

    # y =mx+c, m is slope and c is intercept
    y = (slope * xx) + intercept

    plt.plot(xx, y, 'r')

    # # third
    # xs = [0, -weights[2] / weights[0]]  # x-coordinate of the two points on line.
    # ys = [-weights[2] / weights[1], 0]
    #
    # plt.plot(xs, ys, 'b')
    # plt.savefig(title + '.png')
    plt.pause(interval=.1)


def plot_Perceptron_Delta(inputs, targets, weights_perceptron, weights_delta, title):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=10)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=10)

    plt.title(title)

    plt.ylim(-10, 10)
    plt.xlim(-6, 6)

    xx = np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1]))
    slope = -(weights_perceptron[2] / weights_perceptron[1]) / (weights_perceptron[2] / weights_perceptron[0])
    intercept = -weights_perceptron[2] / weights_perceptron[1]

    # y =mx+c, m is slope and c is intercept
    y = (slope * xx) + intercept

    plt.plot(xx, y, 'y')



    # weight --------------

    xx = np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1]))
    slope = -(weights_delta[2] / weights_delta[1]) / (weights_delta[2] / weights_delta[0])
    intercept = -weights_delta[2] / weights_delta[1]

    # y =mx+c, m is slope and c is intercept
    y = (slope * xx) + intercept

    plt.plot(xx, y, 'c')

    plt.legend(['Perceptron' , 'Delta'], loc='upper right')

    plt.show()



def plot_decision_boundary_mlp(data, targets, mlp, title):
    # Set min and max values and give it some padding
    x_min, x_max = data[0, :].min() - .5, data[0, :].max() + .5
    y_min, y_max = data[1, :].min() - .5, data[1, :].max() + .5
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    samples = np.vstack([xx.ravel(), yy.ravel()]).T

    # Predict the function value for the whole gid

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    boundary = [mlp.predict(sample) for sample in samples]

    boundary = np.reshape(boundary, (xx.shape))

    # Plot the contour and training examples
    plt.contourf(xx, yy, boundary, cmap=plt.cm.Spectral)
    # plt.scatter(data[0, :], data[1, :],  )
    plt.scatter(data[0, idx1], data[1, idx1], s=15, cmap=plt.cm.Spectral)
    plt.scatter(data[0, idx2,], data[1, idx2], s=15, cmap=plt.cm.Spectral)

    plt.title(title)
    plt.show()
if __name__ == "__main__":
    print("Hello World")
