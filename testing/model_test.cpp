//
// Created by malav on 6/23/2022.
//

#include "../classes/Model.hxx"
#include "../classes/loss functions/loss_functions.hxx"
#include "../mnist/read_mnist.hxx"

int main()
{
    Model<CrossEntropy> model;

    double fltr_arr[9] = {0,0,0,0,1,0,0,0,0};
    Mat<double> fltr(3, fltr_arr);

    model.Add<Convolution>(  28    // input width
                           , 28    // input height
                           , fltr // filter
                           , 1    // horizontal stride length
                           , 1    // vertical stride length
                           , true
                           );

    model.Add<MaxPool>(  model.get_outshape(0).width   // input width = 3
                       , model.get_outshape(0).height  // input height = 3
                       , 2  // filter width
                       , 2  // filter height
                       , 1  // horizontal stride length
                       , 1  // vertical stride length
                       );


    model.Add<Linear<RelU>>(  model.get_outshape(1).width * model.get_outshape(1).height  // input size
                            , 10                                                                   // output size
                            );

    model.Add<Softmax>(model.get_outshape(2).width * model.get_outshape(2).height // input size
                        );



//    model.Forward(*input_ptr, output);
//
//    // must call model.Forward(input, output) before calling Backward function because
//    // _local_input member variable must be filled
//    double tar_arr[2] = {1, 0};
//    Vector<double> target(output.get_len(), tar_arr);
//    dLdY = model.get_grad(output, target); hello
//
//    model.Backward(dLdY, dLdX);
//
//    // choose an optimizer
//    Momentum optimizer(model, 0.1, 0.9);
////    SGD optimizer;
//

    DataSet container = read_mnist(10);
    DataSet test_set = read_mnist(100, 1);

    SGD optimizer;

    model.Train(&optimizer, container, 2);
    model.Test(test_set, false);
}

