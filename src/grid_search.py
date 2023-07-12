import pandas as pd
from itertools import product

from model import *
from preprocessing import data_augmentation


def simple_grid_search():
    """A large part of the grid search code is similar to that in the main function and is taken
     from it. Grid search was performed on a reduced dataset due to the time limit of the task."""
    train_val_data = pd.read_csv('../input/train.csv')

    train_data = train_val_data[:5000]
    val_data = train_val_data[5000:6000]

    X_train_orig = np.array(train_data.drop(columns=['label']))
    X_val_orig = np.array(val_data.drop(columns=['label']))
    y_train = np.array(train_data['label'])
    y_val = np.array(val_data['label'])

    X_train_orig, y_train = data_augmentation(X_train_orig, y_train, plots=False)

    X_train = X_train_orig/255
    X_val = X_val_orig/255

    # Grid search was performed using the itertools library. Different value for model size
    # (number of layers and neurons), learning rate for optimizer, alpha parameter of l2
    # regularization, number of epochs and batch size were tested.
    grid_search_list = list(product([0, 1, 2],
                                    [0.01, 0.1],
                                    [1e-5, 1e-3],
                                    [250],
                                    [200, 1000, 2500]))

    best_acc_val, best_hyperparams = 0, None
    for test_hyperparams in grid_search_list:
        model_size = test_hyperparams[0]
        learning_rate = test_hyperparams[1]
        regularizer_alpha = test_hyperparams[2]
        epochs = test_hyperparams[3]
        batch_size = test_hyperparams[4]

        model = SequentialModel()
        if model_size == 0:
            model.add(500, input_dim=28*28, activation='relu')
        elif model_size == 1:
            model.add(800, input_dim=28*28, activation='relu')
            model.add(150, activation='relu')
        elif model_size == 2:
            model.add(1200, input_dim=28*28, activation='relu')
            model.add(700, activation='relu')
            model.add(150, activation='relu')
        model.add(10, activation='softmax')

        model.init_optimizer('RMSprop', learning_rate=learning_rate)
        model.init_regularizer('l2', regularizer_alpha)
        model.set_loss('cross_entropy')
        model.set_metric('accuracy')
        model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, prints_enabled=False)

        # Test (on validation data)
        y_pred = model.predict(X_val)
        loss_val = model.cross_entropy(y_pred, y_val)
        acc_val = model.accuracy(y_pred, y_val)
        print('test values: {}, loss: {:.4}, accuracy: {:.2%}'.format(test_hyperparams, loss_val, acc_val))

        # Check if this combination of hyperparameters is the best so far.
        if acc_val > best_acc_val:
            best_acc_val, best_hyperparams = acc_val, test_hyperparams

    print('Best hyperparameters: {}'.format(best_hyperparams))
