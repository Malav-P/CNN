//
// Created by malav on 6/18/2022.
//

#include "../classes/layers/layer_types.hxx"
#include "../classes/optimizers/SGD.hxx"

int main()
{

    Convolution conv(2,            // number of input maps
                     4,            // input image width
                     4,           // input image height
                     2,           // filter width
                     2,          // filter height
                     1,            // horizontal stride length
                     1,            // vertical stride length
                     false);      // same (true) or valid (false) padding



    conv.get_filter().print();

    std::cout << "\n";

    double in_arr[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    double second_arr[16] = {17, 18, 19, 20 , 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    Mat<double> in(4, 4, in_arr);
    Mat<double> second(4, 4, second_arr);

    in.print();
    second.print();

    std::cout << "\n";

    Vector<double> input = in.flatten();
    Vector<double> second_input = second.flatten();

    std::vector<Vector<double>> inputs(2);
    inputs[0] = input;
    inputs[1] = second_input;

    Vector<double> output(conv.out_shape().height * conv.out_shape().width);

    conv.Forward(inputs, output);

    Mat<double> output_matrix = output.reshape(conv.out_shape().height, conv.out_shape().width);
    output_matrix.print();

    std::cout << "\n";

    Vector<double> dLdY(conv.out_shape().width * conv.out_shape().height);
    dLdY.fill(1);

    std::vector<Vector<double>> dLdX(2);

    conv.Backward(dLdY, dLdX);

    // choose an optimizer
    SGD optimizer(0.1);

    // update parameters for optimizer
    conv.Update_Params(&optimizer,1);


    conv.get_filter().print();
}
