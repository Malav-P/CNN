# CNN
An implementation of a convolutional neural network written in C++.

# Overview

Provided is a header-only library for the implementation of a convolutional neural network. We break down each of the layers later 
in this section. The implementation of a multi-type container class was provided by the [boost](https://www.boost.org/) libraries. The structure
was inspired by mlpack. However, the implementation was done exclusively using my own linear algebra library.

# Layers

The following is a walk-through of how the layers are structured to better help the user understand
how to work with this repository.

## Convolutional Layer

The [Convolutional Layer](./classes/layers/convolution.hxx) file contains the class definition for the Convolutional Layer.
Below is example code of how to construct an instance of this class.

```C++

// constructing a 2 x 2 filter of ones
double fltr_arr[4] = {1,1,1,1};
Mat<double> fltr(2, fltr_arr);

Convolution conv(   28,          // input image width
                    28,          // input image height
                    fltr,        // filter
                    1,           // horizontal stride length
                    1,           // vertical stride length
                    false      // same (true) or valid (false) padding
                );
}

```
## MaxPool Layer

The [MaxPool Layer](./classes/layers/max_pool.hxx) file contains the class definition for a max pooling layer.
Below is example code of how to construct an instance of this class.
```C++
      MaxPool max_pool(   15   // input width
                        , 15   // input height
                        , 2    // window width
                        , 2    // window height
                        , 1    // horizontal stride length
                        , 1    // vertical stride length
                      );

```
## MeanPool Layer

The [MeanPool Layer](./classes/layers/mean_pool.hxx) file contains the class definition for the mean pooling layer.
Below is example code of how to construct an instance of this class.
```C++
    MeanPool mean_pool(   15   // input width
                        , 15   // input height
                        , 2    // window width
                        , 2    // window height
                        , 1    // horizontal stride length
                        , 1    // vertical stride length
                      );

```

## Linear Layer

The [Linear Layer](./classes/layers/linear.hxx) file contains the class definition for the linear layer.
Note that the linear layer is a template class. The template argument when instantiating an object of this 
class is an activation function class. Currently, the source code supports three activation functions : <br\>

- [RelU](./classes/activation%20functions/relU.hxx)
- [Sigmoid](./classes/activation%20functions/sigmoid.hxx)
- [Tanh](./classes/activation%20functions/tanh.hxx)

Below is an example of how to create an instance of the linear layer using the `RelU` activation function class
```C++
   Linear<RelU> linear_layer( 784    // input size
                            , 10     // output size
                            , 0.1    // Leaky RelU parameter
                            );

```


## Output

Currently, the source code supports only a softmax output layer for classification. Below is an example of how
to create an instance of this class
```C++
    Softmax output_layer(10 // input size
                        );


```
# The Model
The [`Model`](./classes/Model.hxx) class provides the class definition for this type. This class 
synthesizes layers and creates a trainable model. Note that this is a template class. The argument to this
template class is a loss function. Currently the source code supports the following loss functions : <br\>

- [Mean Square Error](./classes/loss%20functions/mean_square_error.hxx)
- [Cross Entropy](./classes/loss%20functions/cross_entropy.hxx)

See the example code below for constructing a model using the `CrossEntropy` class.

```C++
    Model<CrossEntropy> model;
    
    // initializing a filter for the convolutional layer
    double fltr_arr[9] = {0,-1,0,-1,5,-1,0,-1,0};
    Mat<double> fltr(3, fltr_arr);
    
    model.Add<Convolution>(   28    // input width
                            , 28    // input height
                            , fltr // filter
                            , 1    // horizontal stride length
                            , 1    // vertical stride length
                            , false
    );
    
    model.Add<MaxPool>(       model.get_outshape(0).width   // input width
                            , model.get_outshape(0).height  // input height
                            , 2  // filter width
                            , 2  // filter height
                            , 1  // horizontal stride length
                            , 1  // vertical stride length
    );
    
    
    model.Add<Linear<RelU>>(model.get_outshape(1).width * model.get_outshape(1).height  // input size
                            , 10   // output size
                            , 0.1  //Leaky RelU parameter
    );
    
    
    
    model.Add<Softmax>(model.get_outshape(2).width * model.get_outshape(2).height // input size
    );
    
    
```

## Training the Model

Training the model can be done by calling the `Train` member function of the `Model` class. It takes the following arguments : <br\>

- An [Optimizer](./classes/optimizers/optimizers.hxx)
  - The source code supports the [Stochastic Gradient Descent](./classes/optimizers/SGD.hxx) and [Momentum](./classes/optimizers/momentum.hxx) gradient descent optimizers
- An object containing the training data. We created our own [data container class](./classes/datasets/dataset.hxx)
- The batch size for updating the weights in the network
- The number of epochs (the number of times to train through the entire dataset)


```C++
model.Train(  &optimizer  // reference to optimizer
            , container   // dataset to train on
            , 50          // batch size
            , N_EPOCHS    // number of epochs
            );  
```