//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MODEL_HXX
#define ANN_MODEL_HXX

#include "layers/layer_types.hxx"
#include "helpers/visitors.hxx"
#include "datasets/dataset.hxx"


template<typename LossFunction>
class Model {
    public:

        // Create a Model object
        Model() = default;

        // Create Model obect from JSON file
        explicit Model(std::string& filename);

        // Destructor to release allocated memory
        ~Model();

        // Add a layer to the model, memory is freed in the destructor ~Model();
        template<typename LayerType, typename... Args>
        void Add(Args... args) {network.push_back(new LayerType(args...));}

        // train the network on the _data
        template<typename Optimizer>
        void Train(Optimizer* opt, DataSet& training_set, size_t batch_size, size_t epochs /* args to be filled */ );

        // return outshape of a layer
        Dims3 get_outshape(size_t idx){ return boost::apply_visitor(Outshape_visitor(), network[idx]);}

        // make a forward pass through the network
        void Forward(Vector<double>& input, Vector<double>& output);

        // make a Backward pass through the network
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // make a pass through the network, updating all the parameters
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer);

        // get number of layers in model
        size_t get_size() const {return network.size();}

        // get const reference to vector of layers
        std::vector<LayerTypes> get_network() const {return network;}

        // test the network
        void Test(DataSet& test_set, bool verbose = false /* args to be filled */ );

        // print model summary
        void print();

        // save model parameters to file
        void save(const std::string& filepath, const std::string& model_name);


    private:
        std::vector<LayerTypes> network;

        LossFunction loss;

};

#include "Model_impl.hxx"
#endif //ANN_MODEL_HXX
