---CODE---
The src folder contains the source code for learning the neural network and 
accompanying functions:
	-data augmentation and preprocessing
	-grid search
	-plot functions
The code consists of defining, training and predicting an MLP neural network, 
and the architecture of the network is built using the numpy library.

---DATA---
The data folder contains input csv files with train and test data:
	-"train.csv"
	-"test.csv"
Train contains 42,000 images of 28x28 dimensions and labels (digits) to which 
they correspond, and the test contains 28,000 images of the same dimensions.
In the data folder is also "best_submission.csv", a file in the format of the 
kaggle competition (https://www.kaggle.com/competitions/digit-recognizer/overview). 
Using this file, an accuracy 98.357% was obtained.

---OTHER---
The Requirements.txt file contains a list of python libraries that need to be 
installed to run the code successfully.