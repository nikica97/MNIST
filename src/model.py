import numpy as np
from math import ceil


class SequentialModel:
    """Neural network model class. It contains all necessary parameters and hyperparameters.
    Vectorized using the NumPy library for faster execution."""

    def __init__(self):
        self.W = []
        self.b = []
        self.s_dw = []  # for RMSProp optimizer
        self.s_db = []  # for RMSProp optimizer
        self.activations = []
        self.learning_rate = None
        self.loss_function = None
        self.metric = None
        self.alpha = None  # regularization
        self.beta = None  # RMSprop
        self.epsilon = None  # RMSprop
        self.labels_num = None
        self.layers_num = 0  # Hidden layers + output layer

    def add(self, units, input_dim=None, activation='linear'):
        """Adds a new fully connected layer.
        :argument:
            units: the number of neurons in the layer
            input_dim: the input dimension for the input layer (number of input parameters)
            activation: the activation function that transforms the values at the neuron's outputs"""

        if input_dim is not None:
            last_layer_size = input_dim
        else:
            last_layer_size = np.shape(self.W[-1])[1]

        self.W.append(
            1e-1*np.random.randn(last_layer_size, units)
        )
        self.b.append(
            np.random.randn(units)
        )
        self.activations.append(activation)
        self.layers_num += 1

    def activation(self, z, activation='linear'):
        """Method for selecting the activation function.
        :argument:
            z: values of neurons in one layer to which the activation function is applied
            activation: name of selected activation function. Possible values: 'linear', 'relu', 'softmax'
        :return:
            values for all neurons after activation function"""

        if activation == 'relu':
            return self.relu(z)
        elif activation == 'softmax':
            return self.softmax(z)
        elif activation == 'linear':
            return z
        else:
            raise Exception('activation {} not implemented. You can select one of the following '
                            'activation functions: "linear", "relu", "softmax".'.format(activation))

    def relu(self, z):
        """Transforms value(s) using function y = max(z, 0).
        :argument:
            z: input vector to transform
        :return:
            transformed vector z"""

        z[z < 0] = 0
        return z

    def softmax(self, z):
        """Transforms value(s) using softmax function.
        :argument:
            z: input vector to transform
        :return:
            transformed vector z"""

        s = np.exp(z)
        total = np.sum(s, axis=1).reshape(-1, 1)
        sigma = s / total
        return sigma

    def set_loss(self, function='cross_entropy'):
        """Method for selecting the loss function which is used in the optimization of neural
        network parameters.
        :argument:
            function: name of selected loss function. Possible values: 'cross_entropy'"""

        if function not in ['cross_entropy']:
            raise Exception('Loss {} not implemented. You can select one of the following '
                            'loss functions: "cross_entropy".'.format(self.loss_function))
        self.loss_function = function

    def set_metric(self, metric='accuracy'):
        """Method for selecting the metric to monitor during learning process.
        :argument:
            metric: name of selected metric. Possible values: 'accuracy'"""

        if metric not in ['accuracy']:
            raise Exception('Metric {} not implemented. You can select one of the following '
                            'metrics: "accuracy".'.format(self.metric))
        self.metric = metric

    def cross_entropy(self, y_pred, y_real):
        """Calculates the cost using the cross entropy function.
        :argument:
            y_pred: model prediction, shape (N, K)
            y_real: ground truth, shape (N, )
        :return:
            average loss"""

        labels = np.arange(self.labels_num)
        y_real_one_hot = np.where(y_real[:,np.newaxis] == labels, True, False)
        all_pred_loss = np.sum(np.log(y_pred) * y_real_one_hot, axis=1)
        return -np.mean(all_pred_loss)

    def accuracy(self, y_pred, y_real):
        """Calculates accuracy as the share of correct predictions in total predictions.
        :argument:
            y_pred: model prediction, shape (N, K)
            y_real: ground truth, shape (N, )
        :return:
            average accuracy"""

        return np.mean(np.argmax(y_pred, axis=1) == y_real)

    def init_optimizer(self, optimizer, learning_rate=5e-1, beta=0.9, epsilon=1e-5):
        """Method for selecting the optimizer, a function that adjusts the attributes of the
        neural network with the aim of faster convergence.
        :argument:
            optimizer: name of selected optimizer. Possible values: 'RMSprop'
            learning_rate: adjustment of parameters speed
            beta: the value of momentum
            epsilon: smoothing term used to prevent division by zero"""

        if optimizer == 'RMSprop':
            self.learning_rate = learning_rate
            self.beta = beta
            self.epsilon = epsilon
            self.s_dw = [1] * self.layers_num
            self.s_db = [1] * self.layers_num
        else:
            raise Exception('optimizer {} not implemented. You can select one of the following '
                            'optimizers: "RMSprop".'.format(optimizer))

    def init_regularizer(self, regularizer, alpha=1e-6):
        """Method for selecting the regularizer, a technique to prevent overfitting.
        :argument:
            regularizer: name of selected regularizer. Possible values: 'l2'
            alpha: coefficient by which the weights within the neural network are multiplied"""

        if regularizer == 'l2':
            self.alpha = alpha
        else:
            raise Exception('regularizer {} not implemented. You can select one of the following '
                            'regularizers: "l2".'.format(regularizer))

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=1000, prints_enabled=True):
        """Method which runs the neural network training process.
        :argument:
            X_train: training data
            y_train: training labels
            X_val: validation data, not mandatory
            y_val: validation labels, not mandatory
            epochs: number of passes through the entire training data
            batch_size: The number of forward-backward propagation passes performed
                        between two refreshes of the neural network parameters
            prints_enabled: variable that determines whether to print loss through the learning process
        :return:
            history: list of loss values and other metrics after each epoch in the learning process"""

        (X, y) = (X_val, y_val) if X_val is not None and y_val is not None else (X_train, y_train)
        history = {'val_loss': [], 'val_acc': []}
        epoch_size = np.shape(X_train)[0]
        self.labels_num = np.shape(self.W[-1])[1]
        for epoch in range(epochs):
            for batch_num in range(ceil(epoch_size/batch_size)):
                batch_start = batch_num*batch_size
                batch_end = batch_start + batch_size
                X_train_batch, y_train_batch = X_train[batch_start:batch_end], y_train[batch_start:batch_end]
                a, z = self.forward_propagation(X_train_batch)
                dW, db = self.backward_propagation(X_train_batch, y_train_batch, a, z)
                self.optimize_weights(dW, db)

            y_pred = self.predict(X)
            loss_val = self.cross_entropy(y_pred, y)
            history['val_loss'].append(loss_val)
            if self.metric == 'accuracy':
                acc_val = self.accuracy(y_pred, y)
                history['val_acc'].append(acc_val)

            if self.metric == 'accuracy' and prints_enabled:
                print('Epoch {}/{} [===========] loss: {:.4} - accuracy: {:.2%}'.format(epoch, epochs, loss_val, acc_val))

        return history

    def forward_propagation(self, X):
        """The method calculates the values of neurons before and after the activation function.
        These values depend on the input X and the current values of weights 'W' and bias 'b'.
        :argument:
            X: input values of the neural network
        :return:
            z: values of neurons before the activation function (not necessary for the input layer)
            a: values of neurons after the activation function
        The value of a[0] also indicates the input of the neural network, and a[-1] is also the output.
        """

        a, z = [None] * (self.layers_num + 1), [None] * self.layers_num

        a[0] = X
        for i, _ in enumerate(self.W):
            z[i] = np.matmul(a[i], self.W[i]) + self.b[i]
            a[i+1] = self.activation(z[i], self.activations[i])

        return a, z

    def predict(self, X):
        """Generates neural network predictions given the input X
        :argument:
            X: input values of the neural network
        :return:
            a[-1]: predictions for every label"""

        a, _ = self.forward_propagation(X)
        return a[-1]

    def backward_propagation(self, X, y, a, z):
        """The procedure for calculating the gradients needed to update the weight parameters of
        the neural network.
        :argument:
            X: input values of the neural network
            y: desired output values of the neural network
            z: values of neurons before the activation function (not necessary for the input layer)
            a: values of neurons after the activation function
        :return:
            dW: values of neurons before the activation function (not necessary for the input layer)
            db: values of neurons after the activation function"""

        delta, grad_W = [None] * self.layers_num, [None] * self.layers_num
        labels = np.arange(self.labels_num)
        instances_num = X.shape[0]

        y_one_hot = np.where(y[:,np.newaxis] == labels, True, False)
        delta[self.layers_num-1] = (a[-1] - y_one_hot)

        for i in range(self.layers_num-1, -1, -1):
            grad_W[i] = np.matmul(a[i].T, delta[i])
            if i > 0:
                if self.activations[i-1] == 'relu':
                    delta[i-1] = np.matmul(delta[i], self.W[i].T) * (z[i-1] > 0)
                elif self.activations[i - 1] == 'linear':
                    delta[i-1] = np.matmul(delta[i], self.W[i].T)

        dW = [grad_W[i] / instances_num + self.alpha * self.W[i] for i in range(self.layers_num)]
        db = [np.mean(delta[i], axis=0) for i in range(self.layers_num)]
        return dW, db

    def optimize_weights(self, dW, db):
        """Refreshing the values of the parameters of the neural network W and b with respect
        to the errors in the previous predictions.
        :argument:
            dW: values of neurons before the activation function (not necessary for the input layer)
            db: values of neurons after the activation function"""

        for i, _ in enumerate(dW):
            self.s_dw[i] = self.beta * self.s_dw[i] + (1 - self.beta) * np.sum(dW[i] ** 2)
            self.W[i] -= self.learning_rate * dW[i] / np.sqrt(self.s_dw[i] + self.epsilon)

            self.s_db[i] = self.beta * self.s_db[i] + (1 - self.beta) * np.sum(db[i] ** 2)
            self.b[i] -= self.learning_rate * db[i] / np.sqrt(self.s_db[i] + self.epsilon)

