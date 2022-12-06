//
// Created by malav on 6/20/2022.
//


#include "../classes/layers/layer_types.hxx"

using namespace CNN;
int main()
{
//    size_t in_width = 4;
//    size_t in_height = 4;
//
//    size_t field_width = 2;
//    size_t field_height = 2;
//
//    double test_input[16] = {0, 67, 7, 937, 45, 9, 34, 87, 4, 56, 8, 4, 9, 72, 73, 60};
//    Vector<> input(in_width*in_height, test_input);
//
//    MeanPool pool_lyr(in_width, in_height, field_width, field_height, 2, 2);
//
//    Vector<> output(pool_lyr.out_shape().width * pool_lyr.out_shape().height);
//
//    pool_lyr.Forward(input, output);
//
//    for (size_t i = 0; i<in_height; i++)
//    {
//        for (size_t j = 0; j<in_width; j++)
//        {
//            std::cout << input[i*in_width + j] << " ";
//        }
//        std::cout << "\n";
//    }
//
//    std::cout << "\n";
//
//    for (size_t i = 0; i<pool_lyr.out_shape().height; i++)
//    {
//        for (size_t j = 0; j<pool_lyr.out_shape().width; j++)
//        {
//            std::cout << output[i*pool_lyr.out_shape().width + j] << " ";
//        }
//        std::cout << "\n";
//    }

    size_t in_width = 4;
    size_t in_height = 4;

    size_t field_width = 2;
    size_t field_height = 2;

    double test_input[16] = {0, 67, 7, 937, 45, 9, 34, 87, 4, 56, 8, 4, 9, 72, 73, 60};
    Vector<double> input(in_width*in_height, test_input);

    double test_input2[16] = {12, 46, 4, 3, 6, 98, 387, 6, 39, 91, 74, 98, 5, 43, 5, 3};
    Vector<double> input2(in_width*in_height, test_input2);

    Vector<double> inputs = input.merge(input2);

    MeanPooling pool_lyr(2, in_width, in_height, field_width, field_height, 2, 2);

    Vector<double> outputs;

    pool_lyr.Forward(inputs, outputs);

    Mat<double> matrix_input(in_height, in_width, inputs.get_data() + 0*in_height*in_width);
    matrix_input.print();

    Mat<double> matrix_output(pool_lyr.out_shape().height, pool_lyr.out_shape().width, outputs.get_data() + 0*pool_lyr.out_shape().width*pool_lyr.out_shape().height);
    matrix_output.print();

    std::cout << "\n";


}
