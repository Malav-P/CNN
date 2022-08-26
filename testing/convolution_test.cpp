//
// Created by malav on 6/18/2022.
//

#include "../classes/layers/convolution.hxx"
#include "../classes/optimizers/SGD.hxx"
#include <iostream>

int main()
{
    double fltr_arr[4] = {1,1,1,1};
    Mat<double> fltr(2, fltr_arr);

    Convolution conv(4, 4, fltr, 1,1,true);

    for (size_t k = 0; k < 4 * conv.get_kernel().get_rows(); k++)
    {
        std::cout << "{ " << (conv.get_indices())[k].width << ", " << (conv.get_indices())[k].height << " }" ;
        std::cout << "\n";
    }

    for (size_t i = 0; i<conv.get_kernel().get_rows(); i++)
    {
        for (size_t j = 0; j<conv.get_kernel().get_cols(); j++)
        {
            std::cout << conv.get_kernel()(i,j) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    double in_arr[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Mat<double> in(4, 4, in_arr);

    for (size_t i = 0; i<in.get_rows(); i++)
    {
        for (size_t j = 0; j<in.get_cols(); j++)
        {
            std::cout << in(i,j) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    Vector<double> input = in.flatten();
    Vector<double> output(conv.out_shape().height * conv.out_shape().width);

    conv.Forward(input, output);

    for (size_t i = 0; i<conv.out_shape().height; i++)
    {
        for (size_t j = 0; j<conv.out_shape().width; j++)
        {
            std::cout << output[conv.out_shape().width*i + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";



    Vector<double> dLdY(conv.out_shape().width * conv.out_shape().height);
    dLdY.fill(1);

    Vector<double> dLdX((conv.in_shape().width) * (conv.in_shape().height));

    conv.Backward(dLdY, dLdX);

    // choose an optimizer
    SGD optimizer(0.1);

    // update parameters for optimizer
    conv.Update_Params(&optimizer,1);

    for (size_t i = 0; i<conv.in_shape().height; i++)
    {
        for (size_t j = 0; j<conv.in_shape().width; j++)
        {
            std::cout << dLdX[conv.out_shape().width*i + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout <<"\n";

    for (size_t i = 0; i<conv.get_kernel().get_rows(); i++)
    {
        for (size_t j = 0; j<conv.get_kernel().get_cols(); j++)
        {
            std::cout << conv.get_kernel()(i,j) << " ";
        }
        std::cout << "\n";
    }
}
