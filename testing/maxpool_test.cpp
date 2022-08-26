//
// Created by malav on 6/20/2022.
//

#include "../classes/layers/max_pool.hxx"
#include <iostream>

int main()
{
    size_t in_width = 4;
    size_t in_height = 4;

    size_t field_width = 2;
    size_t field_height = 2;

    double test_input[16] = {0, 67, 7, 937, 45, 9, 34, 87, 4, 56, 8, 4, 9, 72, 73, 60};
    Vector<> input(in_width*in_height, test_input);

    MaxPool pool_lyr(in_width, in_height, field_width, field_height, 2, 1);

    Vector<> output(pool_lyr.out_shape().width * pool_lyr.out_shape().height);

    pool_lyr.Forward(input, output);

    for (size_t i = 0; i<in_height; i++)
    {
        for (size_t j = 0; j<in_width; j++)
        {
            std::cout << input[i*in_width + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    for (size_t i = 0; i<pool_lyr.out_shape().height; i++)
    {
        for (size_t j = 0; j<pool_lyr.out_shape().width; j++)
        {
            std::cout << output[i*pool_lyr.out_shape().width + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    for (size_t i=0; i<pool_lyr.out_shape().width * pool_lyr.out_shape().height; i++)
    {
        std::cout << pool_lyr.get_winners()[i] << " ";
    }
}