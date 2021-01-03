import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_data = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])
    y_data = np.array([[0.0], [0.36], [0.64], [0.84], [0.96], [1.0], [0.96], [0.84], [0.64], [0.36], [0.0]])

    learning_rate = 0.7
    error = np.zeros((10000, 1))

    # Create 'connection weight'
    weight1 = np.random.random((2, 4))
    weight2 = np.random.random((5, 1))

    #  Layer1 : 11x1 to 11x2
    x_data1 = np.zeros((x_data.shape[0], x_data.shape[1] + 1))
    x_data1[:, :-1] = x_data
    x_data1[:, -1] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    res = np.zeros((11, 5))

    # Iteration
    for k in range(10000):
        # Layer1 : Neural Net
        layer1_1 = np.matmul(x_data1, weight1)

        # Layer1 : Sigmoid
        layer1_2 = 1 / (1 + np.exp(-layer1_1))

        # Layer2 : 11x4 to 11x5
        x_data2 = np.zeros((layer1_2.shape[0], layer1_2.shape[1] + 1))
        x_data2[:, :-1] = layer1_2
        x_data2[:, -1] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # Layer2 : Neural Net
        layer2_1 = np.matmul(x_data2, weight2)

        # Layer2 : Sigmoid
        layer2_2 = 1 / (1 + np.exp(-layer2_1))

        # define 'variable'
        sum_v = np.zeros((11, 5))
        sum_w1n = np.zeros((11, 2))
        sum_w2n = np.zeros((11, 2))
        sum_w3n = np.zeros((11, 2))
        sum_w4n = np.zeros((11, 2))

        # En / Vkj
        for i in range(11):
            sum_v[i] = -(y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i] * x_data2[i]

        # En / W4i
        for i in range(11):
            sum_w4n[i] += np.reshape(-np.reshape(x_data1[i], (2, 1)) * layer1_2[i][3] * (1 - layer1_2[i][3]) * float(weight2[3] * (y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i]), (-1))

        # En / W3i
        for i in range(11):
            sum_w3n[i] += np.reshape(-np.reshape(x_data1[i], (2, 1)) * layer1_2[i][2] * (1 - layer1_2[i][2]) * float(weight2[2] * (y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i]), (-1))

        # En / W2i
        for i in range(11):
            sum_w2n[i] += np.reshape(-np.reshape(x_data1[i], (2, 1)) * layer1_2[i][1] * (1 - layer1_2[i][1]) * float(weight2[1] * (y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i]), (-1))

        # En / W1i
        for i in range(11):
            sum_w1n[i] += np.reshape(-np.reshape(x_data1[i], (2, 1)) * layer1_2[i][0] * (1 - layer1_2[i][0]) * float(weight2[0] * (y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i]), (-1))

        # Update the weights
        for i in range(5):
            weight2[i] -= learning_rate * (sum_v[0][i] + sum_v[1][i] + sum_v[2][i] + sum_v[3][i] + sum_v[4][i] + sum_v[5][i] + sum_v[6][i] + sum_v[7][i] + sum_v[8][i] + sum_v[9][i] + sum_v[10][i])

        for i in range(2):
            weight1[i][0] = weight1[i][0] - learning_rate \
                            * (sum_w1n[0][i] + sum_w1n[1][i] + sum_w1n[2][i] + sum_w1n[3][i] + sum_w1n[4][i] + sum_w1n[5][i] + sum_w1n[6][i] + sum_w1n[7][i] + sum_w1n[8][i] + sum_w1n[9][i] + sum_w1n[10][i])
            weight1[i][1] = weight1[i][1] - learning_rate * (sum_w2n[0][i] + sum_w2n[1][i] + sum_w2n[2][i] + sum_w2n[3][i] + sum_w2n[4][i] + sum_w2n[5][i] + sum_w2n[6][i] + sum_w2n[7][i] + sum_w2n[8][i] + sum_w2n[9][i] + sum_w2n[10][i])
            weight1[i][2] = weight1[i][2] - learning_rate * (sum_w3n[0][i] + sum_w3n[1][i] + sum_w3n[2][i] + sum_w3n[3][i] + sum_w3n[4][i] + sum_w3n[5][i] + sum_w3n[6][i] + sum_w3n[7][i] + sum_w3n[8][i] + sum_w3n[9][i] + sum_w3n[10][i])
            weight1[i][3] = weight1[i][3] - learning_rate * (sum_w4n[0][i] + sum_w4n[1][i] + sum_w4n[2][i] + sum_w4n[3][i] + sum_w4n[4][i] + sum_w4n[5][i] + sum_w4n[6][i] + sum_w4n[7][i] + sum_w4n[8][i] + sum_w4n[9][i] + sum_w4n[10][i])

        for i in range(4):
            error[k] += (y_data[i] - layer2_2[i]) ** 2
        error[k] /= 2

        res = layer2_2

    unlearned_input_data = np.random.uniform(low=0.0, high=1.0, size=110)
    unlearned_input_data = sorted(unlearned_input_data)
    unlearned_input_data = np.reshape(unlearned_input_data, (110, 1))
    unlearned_input_data1 = np.zeros((np.array(unlearned_input_data).shape[0], np.array(unlearned_input_data).shape[1] + 1))
    unlearned_input_data1[:, :-1] = unlearned_input_data
    unlearned_input_data1[:, -1] = [1.0 for _ in range(110)]

    # Layer1 : Neural Net
    layer1_1 = np.matmul(unlearned_input_data1, weight1)

    # Layer1 : Sigmoid
    layer1_2 = 1 / (1 + np.exp(-layer1_1))

    # Layer2 : 11x4 to 11x5
    unlearned_input_data2 = np.zeros((layer1_2.shape[0], layer1_2.shape[1] + 1))
    unlearned_input_data2[:, :-1] = layer1_2
    unlearned_input_data2[:, -1] = [1.0 for _ in range(110)]

    # Layer2 : Neural Net
    layer2_1 = np.matmul(unlearned_input_data2, weight2)

    # Layer2 : Sigmoid
    layer2_2 = 1 / (1 + np.exp(-layer2_1))

    plt.subplot(1, 3, 1)
    plt.plot(error)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.title('Error Graph')

    plt.subplot(1, 3, 2)
    plt.plot(x_data, res)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.title('Graph : 4x(1-x)')

    plt.subplot(1, 3, 3)
    plt.plot(unlearned_input_data, layer2_2)
    plt.xlabel("unlearned data")
    plt.ylabel("output")
    plt.title('Graph of unlearned data')
    plt.show()

