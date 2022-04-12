import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


class neuralNetwork:
    def __init__(self):
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.model = None

    # load train and test dataset
    def load_dataset(self):
        (trainX, trainY), (testX, testY) = mnist.load_data()
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        return trainX, trainY, testX, testY

    # scale images
    def prep_images(self, train, test):
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        return train_norm, test_norm

    # define cnn model
    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    # evaluate a model using k-fold cross-validation
    def evaluate_model(self, dataX, dataY, n_folds=5):
        scores, histories = list(), list()
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(dataX):
            model = self.define_model()
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
            history = model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY), verbose=0)
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('Accuracy = %.3f' % (acc * 100.0))
            scores.append(acc)
            histories.append(history)
        return scores, histories

    # plot diagnostic learning curves
    def summarize_diagnostics(self, histories):
        for i in range(len(histories)):
            plt.subplot(2, 1, 1)
            plt.title('Cross Entropy Loss')
            plt.plot(histories[i].history['loss'], color='blue', label='train')
            plt.plot(histories[i].history['val_loss'], color='orange', label='test')
            plt.subplot(2, 1, 2)
            plt.title('Classification Accuracy')
            plt.plot(histories[i].history['accuracy'], color='blue', label='train')
            plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.show()

    # summarize model performance
    def summarize_performance(self, scores):
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))
        # box and whisker plots of results
        # plt.boxplot(scores)
        # plt.show()

    # run the test harness for evaluating a model
    def run_test_harness(self):
        # load data
        trainX, trainY, testX, testY = self.load_dataset()

        # prepare pixel data
        trainX, testX = self.prep_images(trainX, testX)

        # evaluate model
        scores, histories = self.evaluate_model(trainX, trainY)

        # summarize diagnostics
        self.summarize_diagnostics(histories)

        # summarize estimated performance
        self.summarize_performance(scores)


x = neuralNetwork()
x.run_test_harness()

