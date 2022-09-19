//
// Created by malav on 6/23/2022.
//

#include "../classes/Model.hxx"
#include "../classes/loss functions/loss_functions.hxx"
#include "../mnist/read_mnist.hxx"

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cout << "program takes two arguments\n usage is ./model_test <N_TRAIN> <N_TEST>\n";
    }

    int N_TRAIN = std::stoi(argv[1]);
    int N_TEST = std::stoi(argv[2]);

    if (N_TRAIN < 0)
    {
        std::cout << "usage is ./model_test <N_TRAIN> <N_TEST> where N_TRAIN is a positive integer\n";
        exit(1);
    }

    if (N_TEST < 0)
    {
        std::cout << "usage is ./model_test <N_TRAIN> <N_TEST> where N_TEST is a positive integer\n";
        exit(1);
    }



    Model<CrossEntropy> model;

    double fltr_arr[9] = {1,1,1,1,1,1,1,1,1};
    Mat<double> fltr(3, fltr_arr);

    model.Add<Convolution>(  28    // input width
                           , 28    // input height
                           , fltr // filter
                           , 1    // horizontal stride length
                           , 1    // vertical stride length
                           , false
                           );

    model.Add<MaxPool>(  model.get_outshape(0).width   // input width = 3
                       , model.get_outshape(0).height  // input height = 3
                       , 2  // filter width
                       , 2  // filter height
                       , 1  // horizontal stride length
                       , 1  // vertical stride length
                       );

//    model.Add<Linear<RelU>>(model.get_outshape(1).width * model.get_outshape(1).height  // input size
//                            , 50                                                               // output size
//                            , 0  //Leaky RelU parameter
//                            );


    model.Add<Linear<RelU>>(  model.get_outshape(1).width * model.get_outshape(1).height  // input size
                            , 10                                                                 // output size
                            , 0.1  // leaky RelU parameter
                            );


    model.Add<Softmax>(model.get_outshape(2).width * model.get_outshape(2).height // input size
                        );




    DataSet container = read_mnist(N_TRAIN);
    DataSet test_set = read_mnist(N_TEST, 1);

    //Momentum optimizer(model, 0.1, 0.9);
    SGD optimizer;

    model.Train(&optimizer, container, 50);
    model.Test(test_set, false);

    //! -----------------------------------------------------------------
    char yn;

    std::cout << "Resume program ? y/n \n";
    std::cin >> yn;

    if (yn == 'y')
    {
        std::cout << "Would you like to print the model summary? y/n \n";
        std::cin >> yn;

        if (yn == 'y')
        {
            // print the model summary
            model.print();
        }

        else if (yn == 'n')
        {
            // exit program
            std::cout << "Program exiting ... \n";
            return 0;
        }
    }

    else if (yn == 'n')
    {
        std::cout << "Program exiting ... \n";
        return 0;
    }
}

