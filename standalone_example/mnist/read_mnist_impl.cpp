//
// Created by malav on 7/18/2022.
//

// this file reads in the training data and testing data into formats acceptable by the CNN architecture

#include "read_mnist.hxx"



DataSet read_mnist(size_t N_SAMPLES, size_t train_or_test)
{

    DataSet container({N_SAMPLES, 784});

    FILE *fp;

    int row, col, label;

    char buf[10000];

    char* tok;

    if (train_or_test == 0)
    {fp = fopen("../mnist/mnist_train.csv","r");}

    else {fp = fopen("../mnist/mnist_test.csv","r");}

    if(fp == nullptr) {
        perror("Error opening file \n");
        exit(1);
    }

    for(row=0; row < N_SAMPLES; row++){

        Array<double> mydatapoint({1,784});
        Array<double> mylabel({1,10});

        col = 0;

        fgets(buf, sizeof(buf), fp);

        tok = strtok(buf, ",");
        label = atoi(tok);

        // fill in buffer with correct truth value
        mylabel[{0,label}] = 1.0;

        tok = strtok(nullptr, ",");
        while (tok != nullptr){
            mydatapoint[{0,col}] = strtod(tok, nullptr)/255.0;
            col +=1;
            tok = strtok(nullptr, ",");
        }


        container.datapoints.emplace_back(mydatapoint, mylabel);

    }

    fclose(fp);

    return container;

}


