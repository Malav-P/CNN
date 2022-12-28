//
// Created by malav on 6/23/2022.
//

#include "cnn/Model.hxx"
#include "cnn/datasets/dataset.hxx"
#include "mnist/read_mnist.hxx"

using namespace CNN;

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "program takes three arguments\n usage is ./model_test <N_TRAIN> <N_TEST> <N_epochs>\n";
    }

    int N_TRAIN = std::stoi(argv[1]);
    int N_TEST = std::stoi(argv[2]);
    int N_EPOCHS = std::stoi(argv[3]);

    if (N_TRAIN < 0)
    {
        std::cout << "N_TRAIN must be a positive integer\n";
        exit(1);
    }

    if (N_TEST < 0)
    {
        std::cout << "N_TEST must be a positive integer\n";
        exit(1);
    }

    if (N_EPOCHS <= 0)
    {
        std::cout << "N_EPOCHS must be a positive integer\n";
        exit(1);
    }



    Model<CrossEntropy> model;


    model.Add<Convolution>(  1      // input feature maps
                           , 24    // output feature maps
                           , 28    // input width
                           , 28    // input height
                           , 5     // filter width
                           , 5     // filter height
                           , 1    // horizontal stride length
                           , 1    // vertical stride length
                           , true
                           );

    model.Add<RelU>(    model.get_outshape(0).width,     // input width
                        model.get_outshape(0).height,    // input height
                        model.get_outshape(0).depth,     // input depth
			0.1				 // leaky parameter
    );


    model.Add<MaxPooling>(   model.get_outshape(1).width   // input width
                           , model.get_outshape(1).height  // input height
                           , model.get_outshape(1).depth   // in maps
                           , 2  // filter width
                           , 2  // filter height
                           , 2  // horizontal stride length
                           , 2  // vertical stride length
                           );


    model.Add<Linear>(   model.get_outshape(2).depth
                        *model.get_outshape(2).width
                        *model.get_outshape(2).height  // input size
                        , 256   // output size
                            );

    model.Add<RelU>(    model.get_outshape(3).width,    // input width
                        model.get_outshape(3).height,   // input height
                        model.get_outshape(3).depth,    // input depth
                        0.1                             // leaky RelU parameter
    );

    model.Add<Linear>(   model.get_outshape(4).depth
                         *model.get_outshape(4).width
                         *model.get_outshape(4).height  // input size
                         , 10                           // output size
    );


    model.Add<Softmax>(model.get_outshape(5).width * model.get_outshape(5).height // input size
                        );


    DataSet container = read_mnist(N_TRAIN);
    DataSet test_set = read_mnist(N_TEST, 1);

    SGD optimizer;

    model.Train(&optimizer, container, 50, N_EPOCHS);
    model.Test(test_set, false);

    //model.save("/Users/malavpatel/Coding_Projects/CNN/my_second_CNN.json", "CNN2");

    //Model<CrossEntropy> model2((string&)"/Users/malavpatel/Coding_Projects/CNN/my_second_CNN.json");

    //model2.Test(test_set, false);


    //! -----------------------------------------------------------------
//    char yn;
//
//    std::cout << "Would you like to print the model summary? y/n \n";
//    std::cin >> yn;
//
//    if (yn == 'y')
//    {
//        // print the model summary
//        model.print();
//    }
//
//    else if (yn == 'n')
//    {
//        // exit program
//        std::cout << "Program exiting ... \n";
//        return 0;
//    }

return 0;

}

