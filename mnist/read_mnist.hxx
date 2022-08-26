//
// Created by malav on 7/22/2022.
//

#ifndef ANN_READ_MNIST_HXX
#define ANN_READ_MNIST_HXX

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "../classes/datasets/dataset.hxx"

DataSet read_mnist(size_t N_SAMPLES, size_t train_or_test = 0);

#endif //ANN_READ_MNIST_HXX
