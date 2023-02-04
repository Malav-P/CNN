//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MODEL_HXX
#define ANN_MODEL_HXX

#include "layers/layer_types.hxx"
#include "helpers/visitors.hxx"
#include "datasets/dataset.hxx"
#include "optimizers/optimizers.hxx"
#include "loss_functions/loss_functions.hxx"

#include <fstream>
#include <nlohmann/json.hpp>

namespace CNN {

    /**
     * A class for Models
     *
     * This class is the fundamental structure for a model in this library. The Model class holds the architecture
     * of a model as a vector of layers (more specifically pointers to layers).
     *
     * @tparam LossFunction the loss function used for the Model (current options include MSE and CrossEntropy)
     */
    template<typename LossFunction>
    class Model {
    public:

        /**
         * Default constructor is explicitly defaulted
         */
        Model() = default;

        /**
         * Loads a Model from a JSON file
         * @param filename the absolute filepath to the .json file containing the model parameters
         *
         * @note it is the user's responsibility to choose the correct loss function for the model, it is not written
         * to file. The user is also responsible for loading the correct file. Loading any .json or another file extension
         * will lead to undefined behavior
         */
        explicit Model(std::string &filename);

        /**
         * Destructor for the Model object
         */
        ~Model();

        /**
         * A function to add Layers to the Model architecture
         *
         * @tparam LayerType a valid Layer class, i.e. a layer defined in the layers subdirectory
         * @tparam Args a template parameter pack containing the parameters for adding a Layer to the Model
         * @param args the expanded parameter pack, which is what the constructor for a specific layer uses during construction
         * of an instance of the requested layer
         *
         * @example
         * my_model.Add<Convolution>(arg1, arg2, arg3, ...onwards...)
         */
        template<typename LayerType, typename... Args>
        void Add(Args... args) {network.push_back(new LayerType(args...)); }

        /**
         * Train the Model on a training_set.
         *
         * Execute supervised learning of the Model with (datapoint, label) pairs contained in a training set.
         * @tparam Optimizer the chosen Optimizer class for this model
         * @param opt a pointer to the optimizer used
         * @param training_set the training data, formatted as (datapoint, label) pairs
         * @param batch_size the batch size for training, model parameters are updated after batch_size datapoints have passed through the model
         * @param epochs number of epochs through the dataset
         */
        template<typename Optimizer>
        void Train(Optimizer *opt, DataSet &training_set, size_t batch_size, size_t epochs);

        /**
         * Return the output shape of a specified layer of the network
         * @param idx the index of the requested layer
         * @return a 3-tuple containing the output dimensions of the layer
         */
        Dims3 get_outshape(size_t idx) { return boost::apply_visitor(Outshape_visitor(), network[idx]); }

        /**
         * Propagate a datapoint throughout the model
         *
         * This function iterates through the vector of layers and calls each layer's respective 'Forward' member function
         * @param input the input data
         * @param output the array output data is written to
         */
        void Forward(Array<double> &input, Array<double> &output);

        /**
         * Propagate gradients backwards through the model.
         *
         * This function iterates through the vector of layers and calls each layer's respective 'Backward' member function
         * @param dLdY
         * @param dLdX
         */
        void Backward(Array<double> &dLdY, Array<double> &dLdX);

        /**
         * Update the model's parameters
         *
         * This function iterates through the vector of layers and calls each layer's respective 'Update_Params' member function
         * @tparam Optimizer the Optimizer class used for this model (MSE or CrossEntropy)
         * @param optimizer a pointer to the optimizer object
         * @param normalizer a scaling factor (usually the batch size) given to the model during training
         */
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer, size_t normalizer);

        double Classify(Array<double> & input, Array<double>& output, int& answer);


        /**
         * Utility function to return the vector of layers in the network
         * @return a const reference to the vector of layers
         */
        const std::vector<LayerTypes>& get_network() const { return network; }

        /**
         * Test the model on a training set of (datapoint, label) pairs
         * @param test_set the test set of datapoints and labels
         * @param verbose if true, print the datapoint and the model's output
         */
        void Test(DataSet &test_set, bool verbose = false);

        /**
         * Save the model parameters to a JSON file
         * @param filepath an absolute filepath to the destination
         * @param model_name name for the model
         *
         * @note filepath must have .json extension. Otherwise undefined behavior may occur
         */
        void save(const std::string &filepath, const std::string &model_name);


    private:
        /// vector of layers
        std::vector<LayerTypes> network;

        /// loss function
        LossFunction loss;
    };

}
#include "Model_impl.hxx"
#endif //ANN_MODEL_HXX
