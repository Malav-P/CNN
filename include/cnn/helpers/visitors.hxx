//
// Created by malav on 6/23/2022.
//

// This file defines the classes required for visiting and processing the data in each layer. Since the network
// is an std::vector<LayerTypes> with LayerTypes being a boost::variant type, we define visitors for processing
// the layer.

#ifndef ANN_VISITORS_HXX
#define ANN_VISITORS_HXX

#include <boost/variant.hpp>
#include "pair.hxx"
#include "../lin_alg/data_types.hxx"

namespace CNN {

    // return the output shape of the layer
    class Outshape_visitor : public boost::static_visitor<Dims3> {
    public:

        /**
         * Return the output shape of the layer being visited
         * @tparam T a Layer type, i.e Convolution, Linear, RelU, etc
         * @param operand a pointer to an instance of the above
         * @return the output shape of the Layer
         */
        template<typename T>
        Dims3 operator()(T *operand) const { return operand->out_shape(); }
    };

    // return the input shape of the layer
    class Inshape_visitor : public boost::static_visitor<Dims3> {
    public:

        /**
         * Return the input shape of the layer being visited
         * @tparam T a Layer type, i.e. Convolution, Linear, RelU, etc
         * @param operand a pointer to an instance of the above
         * @return the input shape of the Layer
         */
        template<typename T>
        Dims3 operator()(T *operand) const { return operand->in_shape(); }
    };

    // execute the Forward member function of the layer, using input and output pointers carried by the visitor
    class Forward_visitor : public boost::static_visitor<> {
    public:

        /**
         * Execute the 'Forward' member function of the layer being visited
         * @tparam T a Layer type, i.e. Convolution, Linear, RelU, etc
         * @param operand a pointer to an instance of the above
         */
        template<typename T>
        void operator()(T *operand) const { operand->Forward(*input, *output); }

        Array<double> *input;
        Array<double> *output;
    };

    // execute the Backward member function of the layer, using the dLdY and dLdX pointers carried by the visitor
    class Backward_visitor : public boost::static_visitor<> {
    public:

        /**
         * Execute the 'Backward' member function of the layer being visited
         * @tparam T a Layer type, i.e. Convolution, Linear, RelU, etc
         * @param operand a pointer to an instance of the above
         */
        template<typename T>
        void operator()(T *operand) const { (*operand).Backward(*dLdY, *dLdX); }

        Array<double> *dLdY;
        Array<double> *dLdX;
    };

    // execute the Update_Params member function of the layer, using the specified optimizer and normalizer(usually the batch size)
    // carried by the visitor
    template<typename Optimizer>
    class Update_parameters_visitor : public boost::static_visitor<> {
    public:

        /**
         * Execute the 'Update_Params' member function of the layer being visited using the specified optimizer and normalize
         * @tparam T a Layer type, i.e. Convolution, Linear, RelU, etc
         * @param operand a pointer to an instance of the above
         */
        template<typename T>
        void operator()(T *operand) const { (*operand).Update_Params(optimizer, normalizer); }

        Optimizer *optimizer{nullptr};
        size_t normalizer{1};
    };

} // namespace CNN
#endif //ANN_VISITORS_HXX
