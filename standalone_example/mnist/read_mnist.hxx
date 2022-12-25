//
// Created by malav on 7/22/2022.
//

#ifndef ANN_READ_MNIST_HXX
#define ANN_READ_MNIST_HXX

#include "cnn/classes/datasets/dataset.hxx"
using namespace CNN;

DataSet read_mnist(size_t N_SAMPLES, size_t train_or_test = 0);

#endif //ANN_READ_MNIST_HXX
