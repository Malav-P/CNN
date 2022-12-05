//
// Created by malav on 6/21/2022.
//

#include "../classes/layers/layer_types.hxx"

int main()
{

    size_t in_size = 9;
    size_t out_size = 4;

    double arr[9] = {1, 2, 3, 4, 5 ,6, 7, 8, 9};

    for (size_t i = 0; i< in_size; i++) {std::cout << arr[i] << " ";}

    std::cout << "\n\n";

    Linear linear(in_size, out_size);

    Vector<double> input(in_size, arr);
    Vector<double> output(out_size);

    linear.Forward(input, output);

    std::cout << "\n\n";

    output.print();

    std::cout << "\n\n";


    linear.get_weights().print();

    double out_arr[4] = {4, 2, 2, 4};

    Vector<double> out(4, out_arr);
    Vector<double> dLdX(linear.in_shape().width*linear.in_shape().height);

    linear.Backward(out, dLdX);

}

