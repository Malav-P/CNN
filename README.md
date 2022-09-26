# CNN
An implementation of a convolutional neural network written in C++.

# Overview

Provided is a header-only library for the implementation of a convolutional neural network. We break down each of the layers later 
in this section. The structure for the `Model` class was inspired by [mlpack](https://www.mlpack.org/).
However, the implementation was done exclusively using my own [linear algebra library](./classes/lin_alg).

### Dependencies

- The implementation of a multi-type container class was provided by the [boost](https://www.boost.org/) libraries.
  - If using Linux, run `sudo apt-get install libboost-all-dev` to install the library.
  - If using Windows, it is recommended to use [WSL](https://learn.microsoft.com/en-us/training/modules/get-started-with-windows-subsystem-for-linux/). Then no further action is required
  - If MacOS and using Homebrew package manager, run `brew install boost` to install the library. 


# Layers

The following is a walk-through of how the layers are structured to better help the user understand
how to work with this repository. Eaah layer is its own class.

## `Convolution` 
This class defines the convolutional layer.
The [`convolution.hxx`](./classes/layers/convolution.hxx) file contains the class definition for the Convolutional Layer.
Below is example code of how to construct an instance of this class.

```C++
Convolution conv(   28,          // input image width
                    28,          // input image height
                    3,           // filter width
                    3,           // filter height
                    1,           // horizontal stride length
                    1,           // vertical stride length
                    false        // same (true) or valid (false) padding
                );
```
## `MaxPool` 
This class defines the max pooling layer.
The [`max_pool.hxx`](./classes/layers/max_pool.hxx) file contains the class definition for a max pooling layer.
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
## `MeanPool`
This class defines the mean pooling layer.
The [`mean_pool.hxx`](./classes/layers/mean_pool.hxx) file contains the class definition for the mean pooling layer.
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

## `Linear`

The [`linear.hxx`](./classes/layers/linear.hxx) file contains the class definition for the linear layer.
Note that the linear layer is a template class. The template argument when instantiating an object of this 
class is an activation function class. Currently, the source code supports three activation functions : 

- ReLU activation,  [relU.hxx](./classes/activation%20functions/relU.hxx)
- Sigmoid activation,  [sigmoid.hxx](./classes/activation%20functions/sigmoid.hxx)
- Tanh activation, [tanh.hxx](./classes/activation%20functions/tanh.hxx)

Below is an example of how to create an instance of the linear layer using the `RelU` activation function class
```C++
   Linear<RelU> linear_layer( 784    // input size
                            , 10     // output size
                            , 0.1    // Leaky RelU parameter
                            );
```


## Output

Currently, the source code supports only a softmax output layer for classification. Below is an example of how
to create an instance of this class.
```C++
    Softmax output_layer(10 // input size
                        );
```
# The Model
The [`Model`](./classes/Model.hxx) class provides the class definition for this type. This class 
synthesizes layers and creates a trainable model. Note that this is a template class. The argument to this
template class is a loss function. Currently the source code supports the following loss functions :

- [Mean Square Error](./classes/loss%20functions/mean_square_error.hxx)
- [Cross Entropy](./classes/loss%20functions/cross_entropy.hxx)

See the example code below for constructing a model using the `CrossEntropy` class.

```C++
    Model<CrossEntropy> model;

    
    model.Add<Convolution>(   28      // input width
                            , 28      // input height
                            , 3       // filter width
                            , 3       // filter height
                            , 1       // horizontal stride length
                            , 1       // vertical stride length
                            , false   // same (true) or valid (false) padding
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
                            , 0.1  // Leaky RelU parameter
    );
    
    
    
    model.Add<Softmax>(model.get_outshape(2).width * model.get_outshape(2).height // input size
    );
    
    
```

## Training the Model

Training the model can be done by calling the `Train` member function of the `Model` class. It takes the following arguments :

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