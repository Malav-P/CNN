//
// Created by malav on 6/21/2022.
//

#include "../classes/layers/linear.hxx"
#include "../classes/activation functions/activation_types.hxx"
#include <iostream>

int main()
{

    size_t in_size = 9;
    size_t out_size = 4;

    double arr[9] = {1, 2, 3, 4, 5 ,6, 7, 8, 9};

    for (size_t i = 0; i< in_size; i++) {std::cout << arr[i] << " ";}

    std::cout << "\n\n";

    Linear<Sigmoid> linear(in_size, out_size);

    Vector<double> input(in_size, arr);
    Vector<double> output(out_size);

    linear.Forward(input, output);

    for (size_t i = 0; i<linear.get_local_output().get_len(); i++){
        std::cout << linear.get_local_output()[i] << " ";
    }
    std::cout << "\n\n";

    for (size_t i = 0; i<linear.get_local_output().get_len(); i++){
        std::cout << output[i] << " ";
    }

    std::cout << "\n\n";


    for (size_t i = 0; i<linear.get_weights().get_rows(); i++)
    {
        for (size_t j = 0; j<linear.get_weights().get_cols(); j++)
        {
            std::cout << linear.get_weights()(i,j) << " ";
        }
        std::cout << "\n";
    }

    double out_arr[4] = {4, 2, 2, 4};

    Vector<double> out(4, out_arr);
    Vector<double> dLdX(linear.in_shape().width*linear.in_shape().height);

    linear.Backward(out, dLdX);

}

