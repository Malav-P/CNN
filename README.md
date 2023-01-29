# CNN
An implementation of a convolutional neural network written in C++.

# Overview

Provided is a header-only library for the implementation of a convolutional neural network. We break down each of the layers later 
in this section. The structure for the `Model` class was inspired by [mlpack](https://www.mlpack.org/).
However, the implementation was done using my own [N-D array class](include/cnn/lin_alg/).

### Dependencies

- The implementation of a multi-type container class was provided by the [boost](https://www.boost.org/) libraries.
  - If using Linux, run `sudo apt-get install libboost-all-dev` to install the library.
  - If using Windows, it is recommended to use [WSL](https://learn.microsoft.com/en-us/training/modules/get-started-with-windows-subsystem-for-linux/). Then no further action is required
  - If MacOS and using Homebrew package manager, run `brew install boost` to install the library. 
- Reading saved models is facilitated by a json reading package, [nlohmann_json](https://github.com/nlohmann/json)
  - If MacOS and using Homebrew package manager, run `brew install nlohmann-json` to install the libary

### Installation

To install the package, find your desired working directory and run the following shell command:
```shell
git clone https://github.com/Malav-P/CNN.git
```

Follow the shell commands below to build, make, and install the package at a desired location
```sh
cd CNN # navigate to `CNN` directory
mkdir build # create a new directory named `build`
cd build    # navigate into newly built directory

cmake .. -DCMAKE_INSTALL_PREFIX=<desired installation path> # build project and specify installation path
make  # build CNN library and associated executables, if any
make install # install the library at a desired location 
```

When using this package in your own project, include the following in your `CMakeLists.txt` file:
```
find_package(CNN REQUIRED)
target_link_libraries(<your program here> PRIVATE cnn)
```
and then use the `#include` directive shown below to include the library headers in your project:
```C++
#include "cnn/Model.hxx"
```
When training, the model takes in the training data as a `DataSet` class object. 
If you have data that needs to be placed in a `DataSet` object for the model to train on, you can include the following:
```C++
#include "cnn/datasets/dataset.hxx"
```
NOTE: When building your project, be sure to specify `CMAKE_PREFIX_PATH` variable to correspond to the
`<desired installation path>` that was chosen above. For example, a project that utilizes this library
and has the library installed at `/usr/local/` would call `cmake` like so:
```shell
cmake .. -DCMAKE_PREFIX_PATH=/usr/local
```
# Layers

The following is a walk-through of how the layers are structured to better help the user understand
how to work with this repository. Eaah layer is its own class.

## `Convolution` 
This class defines the convolutional layer.
The [`convolution.hxx`](include/cnn/layers/convolution.hxx) file contains the class definition for the Convolutional Layer.
Below is example code of how to construct an instance of this class.

```C++
    model.Add<Convolution>(   1     // input feature maps
                            , 24    // output feature maps
                            , 28    // input width
                            , 28    // input height
                            , 5     // filter width
                            , 5     // filter height
                            , 1     // horizontal stride length
                            , 1     // vertical stride length
                            , true  // same (true) or valid (false) padding
);
```
## `MaxPooling` 
This class defines the max pooling layer.
The [`max_pooling.hxx`](include/cnn/layers/max_pooling.hxx) file contains the class definition for a max pooling layer.
Below is example code of how to construct an instance of this class.
```C++
    model.Add<MaxPooling>(    model.get_outshape(1).width   // input width
                            , model.get_outshape(1).height  // input height
                            , model.get_outshape(1).depth   // number of input maps
                            , 2  // filter width
                            , 2  // filter height
                            , 2  // horizontal stride length
                            , 2  // vertical stride length
);
```
## `MeanPooling`
This class defines the mean pooling layer.
The [`mean_pooling.hxx`](include/cnn/layers/mean_pooling.hxx) file contains the class definition for the mean pooling layer.
Below is example code of how to construct an instance of this class.
```C++
    model.Add<MeanPooling>(   model.get_outshape(1).width    // input width
                            , model.get_outshape(1).height   // input height
                            , model.get_outshape(1).depth    // number of input maps
                            , 2  // filter width
                            , 2  // filter height
                            , 2  // horizontal stride length
                            , 2  // vertical stride length
);
```

## `Linear`

The [`linear.hxx`](include/cnn/layers/linear.hxx) file contains the class definition for the linear layer. 


Below is an example of how to create an instance of the linear layer using the `RelU` activation function class
```C++
         Linear linear_layer( 784    // input size
                            , 10     // output size
                            );
```

## Activation Functions

Currently the source code supports the following activation functions. 

- [ReLU activation](include/cnn/layers/relU.hxx)
- [Sigmoid activation](include/cnn/layers/sigmoid.hxx)
- [Tanh activation](include/cnn/layers/tanh.hxx)

Below is an example of how to instantiate a member of this class

```C++
     RelU activation( 1         // input width
                    , 380       // input height
                    , 0.1       // leaky parameter
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
The [`Model`](include/cnn/Model.hxx) class provides the class definition for this type. This class 
synthesizes layers and creates a trainable model. Note that this is a template class. The argument to this
template class is a loss function. Currently the source code supports the following loss functions :

- [Mean Square Error](./include/cnn/loss_functions/mean_square_error.hxx)
- [Cross Entropy](./include/cnn/loss_functions/cross_entropy.hxx)

See the example code below for constructing a model using the `CrossEntropy` class.

```C++
Model<CrossEntropy> model;


model.Add<Convolution>(   1     // input feature maps
                        , 24    // output feature maps
                        , 28    // input width
                        , 28    // input height
                        , 5     // filter width
                        , 5     // filter height
                        , 1     // horizontal stride length
                        , 1     // vertical stride length
                        , true
                        );

model.Add<RelU>(    model.get_outshape(0).width,    // input width
                    model.get_outshape(0).height,   // input height
                    model.get_outshape(0).depth,    // input depth
                    0.1                             // leaky RelU parameter
                    );


model.Add<MaxPooling>(  model.get_outshape(1).width   // input width
                      , model.get_outshape(1).height  // input height
                      , model.get_outshape(1).depth   // in maps
                      , 2  // filter width
                      , 2  // filter height
                      , 2  // horizontal stride length
                      , 2  // vertical stride length
                      );


model.Add<Linear>(   model.get_outshape(2).width
                    *model.get_outshape(2).height
                    *model.get_outshape(2).depth // input size
                    , 256   // output size
                    );

model.Add<RelU>(    model.get_outshape(3).width,    // input width
                    model.get_outshape(3).height,   // input height
                    model.get_outshape(3).depth,    // input depth
                    0.1                             // leaky RelU parameter
                    );

model.Add<Linear>(   model.get_outshape(4).width
                    *model.get_outshape(4).height
                    *model.get_outshape(4).depth // input size
                    , 10   // output size
                    );


model.Add<Softmax>(model.get_outshape(5).width * model.get_outshape(5).height // input size
);
```

## Training the Model

Training the model can be done by calling the `Train` member function of the `Model` class. It takes the following arguments :

- An [Optimizer](include/cnn/optimizers/optimizers.hxx)
  - The source code supports the [Stochastic Gradient Descent](include/cnn/optimizers/SGD.hxx)  gradient descent optimizer
- An object containing the training data. We created our own [data container class](include/cnn/datasets/dataset.hxx)
- The batch size for updating the weights in the network
- The number of epochs (the number of times to train through the entire dataset)


```C++
model.Train(  &optimizer  // reference to optimizer
            , container   // dataset to train on
            , 50          // batch size
            , N_EPOCHS    // number of epochs
            );  
```

## Saving and Loading Models

Saving a model can be done by calling the `save` member function. Models are save as `JSON` files.

```C++
std::string filepath = "desired/save/path/my_model.json";
std::string modelname = "your_model_name";        
model.save(filepath, modelname);
```

Loading a model can be done by calling the appropriate constructor on a model file. Input must be `JSON` file. The user
is responsible for determining the `LossFunction` that corresponds to the saved data. 

```C++
std::string filepath = "desired/save/path/my_model.json"
Model<CrossEntropy> model(filepath);
```
