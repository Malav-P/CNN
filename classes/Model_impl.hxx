//
// Created by malav on 6/23/2022.
//

#ifndef ANN_MODEL_IMPL_HXX
#define ANN_MODEL_IMPL_HXX

#include "Model.hxx"

#include "optimizers/optimizers.hxx"
#include "helpers/progress_bar.hxx"
#include <fstream>

using namespace std;

template<typename LossFunction>
void Model<LossFunction>::Forward(Vector<double> &input, Vector<double>& output)
{
    Forward_visitor visitor{};

    // this is here because otherwise leads to memory access violation (SIGTRAP) on line 29
    visitor.input = new Vector<double>(input);


    for (LayerTypes layer : network)
    {
        Dims3 out_shape = boost::apply_visitor(Outshape_visitor(), layer);

        visitor.output = new Vector<double>(out_shape.width*out_shape.height*out_shape.depth);
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

        Dims3 in_shape =  boost::apply_visitor(Inshape_visitor(), layer);

        visitor.dLdX = new Vector<double>(in_shape.width * in_shape.height * in_shape.depth);
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

    // initialize progress bar
    ProgressBar bar;
    bar.set_bar_width(60);
    bar.fill_bar_progress_with("■");
    bar.fill_bar_remainder_with(" ");

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
            if (count % batch_size == 0)
            {
                Update_Params(optimizer, batch_size);
                bar.update(100*(count + i*num_training_points)/(epochs*num_training_points));
            }
        }

        // if remainder exists we can update the model with the remaining datapoints
        if (remainder != 0)
        {
            Update_Params(optimizer, remainder);
            bar.update(100*(count + i*num_training_points)/(epochs*num_training_points));
        }

    }

    std::cout<<"\n";
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

            case 1 : // maxpooling layer
            {
                MaxPooling* ptr = boost::get<MaxPooling*>(layer);

                delete ptr;

                break;
            }

            case 2 : // meanpool layer
            {
                MeanPooling* ptr = boost::get<MeanPooling*>(layer);

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
                ptr->print_filters();
                std::cout << "-------------------------------------------------------------\n\n";

                break;
            }

            case 1 : // maxpooling layer
            {
                std::cout << "-------------------------------------------------------------\n";
                std::cout << "MAXPOOLING Layer, no parameters to be learned. \n\n";
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

template<typename LossFunction>
void Model<LossFunction>::save(const string& filepath, const string& model_name)
{

    char buf[1000];

    ofstream fid(filepath);

    // if file fails to open, exit program
    if (!fid)
    {
        exit(1);
    }

    // print model name to json file
    fid << "{\n \"" << model_name << "\" : [\n";

    std::cout << "\nSaving Model...\n";
    fid << buf;


    size_t count = 0;
    for (LayerTypes layer: network)
    {
        count +=1;
        switch (layer.which())
        {
            case 0 : // convolutional layer
            {

                Convolution* ptr = boost::get<Convolution*>(layer);

                size_t layerID = 0;
                size_t parameters[12];
                double* weights;

                // in maps
                parameters[0] = ptr->in_shape().depth;
                // out maps
                parameters[1] = ptr->out_shape().depth;
                // in width
                parameters[2] = ptr->in_shape().width;
                // out width
                parameters[3] = ptr->in_shape().height;
                // filter width
                parameters[4] = ptr->get_filters()[0].get_cols();
                // filter height
                parameters[5] = ptr->get_filters()[0].get_rows();
                // horizontal stride
                parameters[6] = ptr->get_stride().width;
                // vertical stride
                parameters[7] = ptr->get_stride().height;
                // pad left
                parameters[8] = ptr->get_padding().first;
                // pad right
                parameters[9] = ptr->get_padding().second;
                // pad top
                parameters[10] = ptr->get_padding().third;
                // pad bottom
                parameters[11] = ptr->get_padding().fourth;

                // filters
                std::vector<Cuboid<double>> _filters = ptr->get_filters();


                size_t length = _filters[0].get_rows() *_filters[0].get_cols() * _filters[0].get_depth();
                // linear array of weights
                weights = new double[length * parameters[1]];
                size_t offset;
                for (size_t l = 0; l < parameters[1]; l++)
                {
                    offset =  length * l;
                    std::memcpy(weights + offset, _filters[l].get_data(), length * sizeof(double));

                }


                fid << "\t{\n\t \"layerID\" : " << layerID << ", \n\t \"parameters\" : [";

                // write parameters
                for (size_t i = 0; i<11; i++)
                {
                    fid << parameters[i] <<",";
                }
                fid << parameters[11] << "], ";

                // write weights
                fid << "\n\t \"weights\" : [";
                for (size_t i = 0; i < length * parameters[1] - 1; i++)
                {
                    fid << weights[i] << ",";
                }
                fid << weights[length * parameters[1] - 1] << "] \n\t}";

                delete[] weights;

                break;
            }

            case 1 : // maxpooling layer
            {

                MaxPooling* ptr = boost::get<MaxPooling*>(layer);

                size_t layerID = 1;
                size_t parameters[7];

                // in maps
                parameters[0] = ptr->in_shape().depth;
                // in width
                parameters[1] = ptr->in_shape().width;
                // in height
                parameters[2] = ptr->in_shape().height;
                // field width
                parameters[3] = ptr->get_field().width;
                // field height
                parameters[4] = ptr->get_field().height;
                // hor stride
                parameters[5] = ptr->get_stride().width;
                // v stride
                parameters[6] = ptr->get_stride().height;

                fid << "\t{\n\t \"layerID\" : " << layerID << ", \n\t \"parameters\" : [";

                // write parameters
                for (size_t i = 0; i<6; i++)
                {
                    fid << parameters[i] <<",";
                }
                fid << parameters[6] << "], ";

                // write weights
                fid << "\n\t \"weights\" : null \n\t}";

                break;
            }

            case 2 : // meanpooling layer
            {

                MeanPooling* ptr = boost::get<MeanPooling*>(layer);

                size_t layerID = 2;
                size_t parameters[7];

                // in maps
                parameters[0] = ptr->in_shape().depth;
                // in width
                parameters[1] = ptr->in_shape().width;
                // in height
                parameters[2] = ptr->in_shape().height;
                // field width
                parameters[3] = ptr->get_field().width;
                // field height
                parameters[4] = ptr->get_field().height;
                // hor stride
                parameters[5] = ptr->get_stride().width;
                // v stride
                parameters[6] = ptr->get_stride().height;

                fid << "\t{\n\t \"layerID\" : " << layerID << ", \n\t \"parameters\" : [";

                // write parameters
                for (size_t i = 0; i<6; i++)
                {
                    fid << parameters[i] <<",";
                }
                fid << parameters[6] << "], ";

                // write weights
                fid << "\n\t \"weights\" : null \n\t}";

                break;
            }

            case 3 : // Linear layer
            {
                Linear* ptr = boost::get<Linear*>(layer);

                size_t layerID = 3;
                size_t in_size = ptr->in_shape().height;
                size_t out_size = ptr->out_shape().height;
                double* weights = ptr->get_weights().get_data();

                // write parameters
                fid << "\t{\n\t \"layerID\" : " << layerID << ", \n\t \"parameters\" : [" << in_size << "," << out_size << "], ";

                // write weights
                fid << "\n\t \"weights\" : [";
                for (size_t i = 0; i < in_size*out_size - 1; i++)
                {
                    fid << weights[i] << ",";
                }
                fid << weights[in_size*out_size - 1] << "] \n\t}";

                break;
            }


            case 4 : // Softmax layer
            {
                Softmax* ptr = boost::get<Softmax*>(layer);

                size_t layerID = 4;
                size_t in_size = ptr->in_shape().height;
                double temperature = ptr->get_beta();

                // write parameters
                fid << "\t{\n\t \"layerID\" : " << layerID << ", \n\t \"parameters\" : [" << in_size << "," << temperature <<  "], ";

                // write weights
                fid << "\n\t \"weights\" : null \n\t}";

                break;
            }

            case 5 : // RelU layer
            {
                RelU* ptr = boost::get<RelU*>(layer);

                size_t layerID = 5;
                double leaky_param = ptr->get_leaky_param();
                size_t in_width = ptr->in_shape().width;
                size_t in_height = ptr->in_shape().height;
                size_t in_depth = ptr->in_shape().depth;

                fid << "\t{\n\t \"layerID\" : " << layerID << ", \n\t \"parameters\" : [" << leaky_param << "," << in_width << "," << in_height<< "," << in_depth << "], ";

                // write weights
                fid << "\n\t \"weights\" : null \n\t}";

                break;
            }

            case 6: // Sigmoid Layer
            {


                break;
            }
            case 7: // Tanh Layer
            {


                break;
            }

            default : // nothing to do
            {
                break;
            }
        }

        if (count < network.size())
        {
            fid << ",\n";
        }
    }

    fid << "\n]\n}";

    fid.close();

}

#endif //ANN_MODEL_IMPL_HXX
