import numpy as np
import pandas as pd
from sklearn import datasets

def iris_dataset_prepocessing():
    iris = datasets.load_iris()
    data = iris['data']
    target = iris['target']
    target_names = iris['target_names']
    target_names_dic = {}

    for each in target_names:
        target_names_dic[each] = []

    df = pd.DataFrame({'target': target,
                       's_length': data[:, 0],
                       's_width': data[:, 1],
                       'p_length': data[:, 2],
                       'p_width': data[:, 3]
                       })

    for index, row in df.iterrows():
        if row['target'] == 0:
            target_names_dic[target_names[0]].append(1)
            target_names_dic[target_names[1]].append(0)
            target_names_dic[target_names[2]].append(0)
        elif row['target'] == 1:
            target_names_dic[target_names[0]].append(0)
            target_names_dic[target_names[1]].append(1)
            target_names_dic[target_names[2]].append(0)
        elif row['target'] == 2:
            target_names_dic[target_names[0]].append(0)
            target_names_dic[target_names[1]].append(0)
            target_names_dic[target_names[2]].append(1)

    for each in target_names_dic:
        df[each] = target_names_dic[each]

    df.drop(["target"], axis=1, inplace=True)

    return df

class NeuralNetwork:
    def __init__(self, x_train, y_train):
        self.X = x_train
        self.y = y_train
        self.theta_1 = np.random.normal(0, 0.1, size=(4, 4))
        self.theta_2 = np.random.normal(0, 0.1, size=(3, 4))

    def softmax_activation(self, z):
        exp_z = np.exp(z)
        sum_z = np.sum(exp_z)
        normalized = exp_z / sum_z

        return normalized

    def forward_propagate(self, x):
        # first layer activation
        z_2 = np.matmul(self.theta_1, x)
        a_2 = self.softmax_activation(z_2)

        # second layer activation
        z_3 = np.matmul(self.theta_2, a_2)
        a_3 = self.softmax_activation(z_3)

        return a_2, a_3

    def backpropagation(self):
        DELTA_1 = np.zeros((4, 4))
        DELTA_2 = np.zeros((3, 4))
        for i, xi in enumerate(self.X):
            a2, a3 = self.forward_propagate(xi)
            delta_3 = a3 - self.y[i]
            delta_2 = np.matmul(np.transpose(self.theta_2), delta_3) * a2 * (1-a2)

            DELTA_1 = DELTA_1 + np.matmul(delta_2.reshape(4, 1), xi.reshape(1, 4))
            DELTA_2 = DELTA_2 + np.matmul(delta_3.reshape(3, 1), a2.reshape(1, 4))
        m = self.X.shape[0]
        D_1 = DELTA_1 / m
        D_2 = DELTA_2 / m

        return D_1, D_2



data = iris_dataset_prepocessing()
X = data[['s_length', 's_width', 'p_length', 'p_width']].to_numpy()
y = data[['setosa', 'versicolor', 'virginica']].to_numpy()

nn = NeuralNetwork(X, y)
print(nn.backpropagation()[1])