//
// Created by malav on 6/23/2022.
//

#ifndef ANN_MODEL_IMPL_HXX
#define ANN_MODEL_IMPL_HXX

#include "Model.hxx"

#include "optimizers/optimizers.hxx"

template<typename LossFunction>
void Model<LossFunction>::Forward(Vector<double> &input, Vector<double>& output)
{
    Forward_visitor visitor{};

    // this is here because otherwise leads to memory access violation (SIGTRAP) on line 29
    visitor.input = new Vector<double>(input);


    for (LayerTypes layer : network)
    {
        Dims out_shape = boost::apply_visitor(Outshape_visitor(), layer);

        visitor.output = new Vector<double>(out_shape.width*out_shape.height);
        boost::apply_visitor(visitor, layer);

        delete visitor.input;
        visitor.input = visitor.output;
    }

    // this uses copy assignment operator, need to deallocate memory associated with pointer visitor.output, which is
    // done in the line below
    output = *(visitor.output);

    delete visitor.output;
}

template<typename LossFunction>
void Model<LossFunction>::Backward(Vector<double> &dLdY, Vector<double>& dLdX)
{
    Backward_visitor visitor{};

    // this is here because otherwise leads to memory access violation (SIGTRAP) on line 54
    visitor.dLdY = new Vector<double>(dLdY);

    // i must be type int or else code fails! i-- turn i = 0 into i = largest unsigned int possible
    for (int i = network.size() - 1; i >= 0; i--)
    {
        LayerTypes layer = network[i];

        Dims in_shape =  boost::apply_visitor(Inshape_visitor(), layer);

        visitor.dLdX = new Vector<double>(in_shape.width * in_shape.height);
        boost::apply_visitor(visitor, layer);

        delete visitor.dLdY;

        visitor.dLdY = visitor.dLdX;
    }

    // this uses copy assignment operator, need to deallocate memory associated with pointer visitor.dLdX, which is
    // done in the line below
    dLdX = *(visitor.dLdX);

    delete visitor.dLdX;
}

template<typename LossFunction>
template<typename Optimizer>
void Model<LossFunction>::Update_Params(Optimizer* optimizer, size_t normalizer)
{
    // create visitor object
    Update_parameters_visitor<Optimizer> visitor {};

    // give visitor the optimizer and normalizer values
    visitor.normalizer = normalizer;
    visitor.optimizer = optimizer;

    // send visitor to each layer to update weights and biases
    for (LayerTypes layer : network)
    {
        boost::apply_visitor(visitor, layer);
    }

    // reset the optimizer for the next pass through the network
    (*optimizer).reset();
}


template<typename LossFunction>
template<typename Optimizer>
void Model<LossFunction>::Train(Optimizer* optimizer, DataSet& training_set, size_t batch_size, size_t epochs /* args TBD */)
{

    // determine if number of training points is divisible by the batch_size
    //      - if there is no remainder, we will be updating the parameters (num training points) / (batch_size) times
    //      - if there is a remainder, we will update the parameters |_ (num training points) / (batch_size) _| times
    //        and then proceed to train on the remainder of the training set ( num training points % batch_size)


    size_t num_training_points = training_set.shape.width;

    size_t remainder = num_training_points % batch_size;

    // for each data point in my training set:
    //      - make a Forward pass
    //      - compute dLdY, the loss gradient at the output layer as a result of this Forward pass
    //      - make a Backward pass, backpropagating the calculated loss gradient
    //      - if we have sent a batch_size amount of data points forward and backward after this last pass, update the
    //        parameters in the networks using Update_Params function

    for (size_t i = 0; i < epochs ; i++) {

        Vector<double> output, dLdY, dLdX;
        size_t count = 0;

        for (Vector_Pair datapoint: training_set.datapoints) {
            // make forward pass, datapoint.first is the input
            Forward((datapoint.first), output);

            // compute dLdY, datapoint.second is the label
            dLdY = loss.grad(output, (datapoint.second));


            // make a backward pass
            Backward(dLdY, dLdX);
            count += 1;

            // update parameters if we have propagated batch_size number of samples
            if (count % batch_size == 0) { Update_Params(optimizer, batch_size); }
        }

        // if remainder exists we can update the model with the remaining datapoints
        if (remainder != 0) { Update_Params(optimizer, remainder); }

    }

}

template<typename LossFunction>
void Model<LossFunction>::Test(DataSet &test_set, bool verbose)
{
    Vector<double> output, dLdY, dLdX;
    size_t num_correct = 0;
    for (Vector_Pair datapoint : test_set.datapoints)
    {
        // make forward pass, datapoint.first is the input
        Forward((datapoint.first), output);

        // find index with largest probability
        int idx = 0;
        for (int i = 0; i< output.get_len(); i++)
        {
            if (output[idx] < output[i])
            { idx = i; }
        }

        // check if idx is correct with the label
        if (datapoint.second[idx] == 1)
        {
            num_correct += 1;
        }

        if (verbose)
        {
            // print output vector
            output.print();

            // print target vector
            datapoint.second.print();
        }

    }

    std::cout << "model got " << num_correct << " guesses correct out of " << test_set.datapoints.size() << " samples.\n";
}


//destructor
template<typename LossFunction>
Model<LossFunction>::~Model()
{
    if (network.empty())
    {
        // do nothing
        return;
    }

    // free memory that was allocated using model.Add() function
    for (LayerTypes layer: network)
    {
        switch (layer.which())
        {
            case 0 : // convolutional layer
            {
                Convolution* ptr = boost::get<Convolution*>(layer);

                delete ptr;

                break;
            }

            case 1 : // maxpool layer
            {
                MaxPool* ptr = boost::get<MaxPool*>(layer);

                delete ptr;

                break;
            }

            case 2 : // meanpool layer
            {
                MeanPool* ptr = boost::get<MeanPool*>(layer);

                delete ptr;

                break;
            }

            case 3 : // Linear layer
            {
                Linear* ptr = boost::get<Linear*>(layer);

                delete ptr;

                break;
            }


            case 4 : // Softmax layer
            {
                Softmax* ptr = boost::get<Softmax*>(layer);

                delete ptr;

                break;
            }

            case 5: // RelU layer
            {
                RelU* ptr = boost::get<RelU*>(layer);

                delete ptr;

                break;
            }

            case 6: // Sigmoid Layer
            {
                Sigmoid* ptr = boost::get<Sigmoid*>(layer);

                delete ptr;

                break;
            }
            case 7: // Tanh Layer
            {
                Tanh* ptr = boost::get<Tanh*>(layer);

                delete ptr;

                break;
            }


            default : // nothing to do
            {
                break;
            }
        }
    }


}

template<typename LossFunction>
void Model<LossFunction>::print()
{
    for (LayerTypes layer: network)
    {
        switch (layer.which())
        {
            case 0 : // convolutional layer
            {
                Convolution* ptr = boost::get<Convolution*>(layer);
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "CONV Layer \n The filter is shown below:\n\n";
                ptr->get_filter().print();
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            case 1 : // maxpool layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "MAXPOOL Layer, no parameters to be learned. \n\n";
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            case 2 : // meanpool layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "MEANPOOL Layer, no parameters to be learned. \n\n";
                std::cout << "-------------------------------------------------------------\n\n";
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            case 3 : // Linear layer
            {
                Linear* ptr = boost::get<Linear*>(layer);
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "This was a Linear Layer\n";
                std::cout << "The weight matrix is : \n \n";
                //ptr->get_weights().print();
                std::cout << "The bias vector is  : \n \n";
                //ptr->get_biases().print();
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }


            case 4 : // Softmax layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "This was a Softmax Layer, no parameters are to be learned. \n\n";
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            case 5 : // RelU layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "This was a RelU Layer, no parameters are to be learned. \n\n";
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            case 6: // Sigmoid Layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "This was a Sigmoid Layer, no parameters are to be learned. \n\n";
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }
            case 7: // Tanh Layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "This was a Tanh Layer, no parameters are to be learned. \n\n";
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            default : // nothing to do
            {
                break;
            }
        }
    }

}

#endif //ANN_MODEL_IMPL_HXX
