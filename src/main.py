from model import *
from preprocessing import data_augmentation
from plot_functions import *


def main():
    # Read the data from input files
    train_val_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv("../data/test.csv")

    # Split data to train and validation data
    train_data = train_val_data[:35000]  # train_data = train_val_data[:42000] for final training
    val_data = train_val_data[35000:42000]

    # Split data to inputs X (images) and outputs y (labels)
    X_train_orig = np.array(train_data.drop(columns=['label']))
    X_val_orig = np.array(val_data.drop(columns=['label']))
    X_test_orig = np.array(test_data)
    y_train = np.array(train_data['label'])
    y_val = np.array(val_data['label'])

    #Show few sample images
    #for i in range(5):
    #    show_sample(X_train_orig[i])

    # Compare the number of examples for different labels
    print_label_counters(y_train)

    # Generate new images for the dataset created by modifying existing images
    X_train_orig, y_train = data_augmentation(X_train_orig, y_train, plots=False)

    # Scale data (All pixel values are from 0 to 255, so there is no need for more complex scalers)
    X_train = X_train_orig/255
    X_val = X_val_orig/255
    X_test = X_test_orig/255

    # Define the model and add neuron layers along with activation functions.
    # Since the input images are 28x28, which is the typical size after convolution and pooling layers
    # in CNN, there is no crucial need for CNN, and more focus is placed on the data augmentation part.
    # Therefore, a simpler MLP architecture is used here.
    model = SequentialModel()
    model.add(800, input_dim=28*28, activation='relu')
    model.add(150, activation='relu')
    model.add(10, activation='softmax')

    # Here, additional hyperparameters required for the learning process are given to the neural network.
    model.init_optimizer('RMSprop', learning_rate=0.1, beta=0.99)  # rekord 3e-1
    model.init_regularizer('l2', 1e-3)
    model.set_loss('cross_entropy')
    model.set_metric('accuracy')

    # Train neural network weight parameters
    history = model.fit(X_train, y_train, X_val, y_val, epochs=250, batch_size=1000)
    plot_history(history)

    # Test predictions accuracy (on validation data)
    y_pred = model.predict(X_val)
    loss_val = model.cross_entropy(y_pred, y_val)
    acc_by_sample = np.argmax(y_pred, axis=1) == y_val
    acc_val = np.mean(acc_by_sample)
    print('Validation set final cross entropy loss: {:.4}, final accuracy: {:.2%}'.format(loss_val, acc_val))

    #Check confusion matrix
    show_confusion_matrix(y_pred, y_val)

    # Show images for which the model gave wrong predictions
    for i, acc in enumerate(acc_by_sample[:100]):
        if not acc:
            show_sample(X_val[i], title='Real value: {}, predicted value: {}'.format(y_val[i], np.argmax(y_pred[i])))

    # Generate predictions for test data (submission achieved kaggle score: 0.98375)
    y_pred = model.predict(X_test)
    df_output = pd.DataFrame(np.argmax(y_pred, axis=1), columns=['Label'])
    df_output.insert(loc=0, column='ImageId', value=np.arange(len(y_pred)) + 1)
    df_output.to_csv("../data/submission.csv", index=False)


if __name__ == '__main__':
    main()
    #simple_grid_search()
