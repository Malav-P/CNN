//
// Created by malav on 7/3/2022.
//

#ifndef ANN_MOMENTUM_H
#define ANN_MOMENTUM_H

#include "../lin_alg/data_types.hxx"
#include "../Model.hxx"


// momentum
class Momentum {

public:

    // constructor
    Momentum() = default;

    // constructor
    template <typename L>
    Momentum(Model<L>& model, double learn_rate, double beta)
            : alpha(learn_rate),
              beta(beta),
              velocities_mat(new Mat<double>*[model.get_size()]),
              velocities_vec(new Vector<double>*[model.get_size()]),
              model_size(model.get_size())
    {
        std::vector<LayerTypes> network = model.get_network();
        LayerTypes layer;
        for (size_t i = 0; i < model_size; i++)
        {
            layer = network[i];
            switch (layer.which())
            {
                case 0 : // convolutional layer
                {
                    size_t rows = boost::get<Convolution*>(layer)->get_kernel().get_rows();
                    size_t cols = boost::get<Convolution*>(layer)->get_kernel().get_cols();

                    velocities_mat[i] = new Mat<double>(rows, cols);
                    velocities_vec[i] = nullptr;

                    break;
                }

                case 3 : // Linear<RelU> layer
                {
                    size_t rows = boost::get<Linear<RelU>*>(layer)->get_weights().get_rows();
                    size_t cols = boost::get<Linear<RelU>*>(layer)->get_weights().get_cols();

                    velocities_mat[i] = new Mat<double>(rows, cols);
                    velocities_vec[i] = new Vector<double>(rows);

                    break;
                }

                case 4 : // Linear<Sigmoid> layer
                {
                    size_t rows = boost::get<Linear<Sigmoid>*>(layer)->get_weights().get_rows();
                    size_t cols = boost::get<Linear<Sigmoid>*>(layer)->get_weights().get_cols();

                    velocities_mat[i] = new Mat<double>(rows, cols);
                    velocities_vec[i] = new Vector<double>(rows);

                    break;
                }

                case 5 : // Linear<Tanh> layer
                {
                    size_t rows = boost::get<Linear<Tanh>*>(layer)->get_weights().get_rows();
                    size_t cols = boost::get<Linear<Tanh>*>(layer)->get_weights().get_cols();

                    velocities_mat[i] = new Mat<double>(rows, cols);
                    velocities_vec[i] = new Vector<double>(rows);

                    break;
                }

                default : // other layers that dont have weights or biases to train
                {
                    velocities_mat[i] = nullptr;
                    velocities_vec[i] = nullptr;

                    break;
                }
            }
        }
    }

    // destructor
    ~Momentum()
    {
        for (size_t i = 0; i < model_size ; i++)
        {
            // free the pointers in the array
            delete velocities_mat[i];
            delete velocities_vec[i];
        }

        // free the pointer to array of pointer
        delete[] velocities_mat;
        delete[] velocities_vec;
    }

    // apply optimizer to matrix objects
    void Forward(Mat<double>& weights, Mat<double>& gradient, size_t normalizer)
    {
        // velocity = beta * velocity + (1 - beta) * (1/normalizer) * gradient
        // this routine is inefficient with operator overloads since multiple for loops will be called for each
        // operator overload. Consider using a single loop over the indices i, j to complete this task
        (*velocities_mat[k]) = ((*velocities_mat[k]) * beta) + (gradient * ((1.0 - beta) * (1.0/normalizer)));

        // update weights
        weights += (*velocities_mat[k]) * (-alpha);

        // move the index k to the next nonnull layer in the network
        k+=1;
        while (k<model_size && velocities_mat[k] == nullptr) {k++;}
    }

    // biases = biases - alpha * gradient
    void Forward(Vector<double>& biases, Vector<double>& gradient, size_t normalizer)
    {
        {
            // velocity = beta * velocity + (1 - beta) * (1/normalizer) * gradient
            // this routine is inefficient with operator overloads since multiple for loops will be called for each
            // operator overload. Consider using a single loop over the indices i, j to complete this task
            (*velocities_vec[k]) = ((*velocities_vec[k]) * beta) + (gradient * ((1.0 - beta) * (1.0/normalizer)));

            // update biases
            biases += (*velocities_vec[k]) * (-alpha);
        }
    }

    // reset the layer counter of the optimizer
    void reset() { k = 0;}

    // for debugging
    void print_vel_mat()
    {
        for (size_t i = 0; i <model_size; i++)
        {
            if (velocities_mat[i] == nullptr) { std::cout << "nullptr, ";}
            else {std::cout << "{ " << (*velocities_mat[i]).get_rows() << ", " << (*velocities_mat[i]).get_cols() << " }, "; }
        }

        std::cout << "\n";
    }

    // for debugging
    void print_vel_vec()
    {
        for (size_t i = 0; i <model_size; i++)
        {
            if (velocities_vec[i] == nullptr) { std::cout << "nullptr, ";}
            else {std::cout << "{ " << (*velocities_vec[i]).get_len() << " }, "; }
        }

        std::cout << "\n";
    }

private:

    // learning rate
    double alpha {0.1};

    // exponential average hyperparameter
    double beta {0.9};

    // pointer to an array of pointers of velocities for matrix objects
    Mat<double>** velocities_mat {nullptr};

    // pointer to an array of pointers of velocities for vector objects
    Vector<double>** velocities_vec {nullptr};

    // size of model
    size_t model_size {0};

    // current layer that the optimizer is working on
    size_t k {0};
};
#endif //ANN_MOMENTUM_H
