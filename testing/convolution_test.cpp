//
// Created by malav on 6/18/2022.
//

#include "../classes/layers/layer_types.hxx"
#include "../classes/optimizers/SGD.hxx"

int main()
{

    Convolution conv(4,            // input image width
                     4,           // input image height
                     2,           // filter width
                     2,          // filter height
                     1,            // horizontal stride length
                     1,            // vertical stride length
                     false);      // same (true) or valid (false) padding



    conv.get_filter().print();

    std::cout << "\n";

    double in_arr[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Mat<double> in(4, 4, in_arr);

    in.print();

    std::cout << "\n";

    Vector<double> input = in.flatten();
    Vector<double> output(conv.out_shape().height * conv.out_shape().width);

    conv.Forward(input, output);

    conv.get_local_input().print();

    Mat<double> output_matrix = output.reshape(conv.out_shape().height, conv.out_shape().width);
    output_matrix.print();

    std::cout << "\n";



    Vector<double> dLdY(conv.out_shape().width * conv.out_shape().height);
    dLdY.fill(1);

    Vector<double> dLdX(in.get_cols() * in.get_rows());

    conv.Backward(dLdY, dLdX);

    // choose an optimizer
    SGD optimizer(0.1);

    // update parameters for optimizer
    conv.Update_Params(&optimizer,1);

    for (size_t i = 0; i<in.get_rows(); i++)
    {
        for (size_t j = 0; j<in.get_cols(); j++)
        {
            std::cout << dLdX[in.get_cols()*i + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout <<"\n";

    conv.get_filter().print();
}
