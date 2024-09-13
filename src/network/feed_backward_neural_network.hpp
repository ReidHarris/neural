/*
 * bias_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_FEED_BACKWARD_HPP_
#define SRC_FEED_BACKWARD_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>

#include "plain_neural_network.hpp"

namespace neural
{


template <typename... Layers>
class FeedbackwardNeuralNetwork
{
public :
	template <size_t N> using LayerType = std::tuple_element<N, std::tuple<Layers...>>::type;
	template <size_t N> using LWeightType = LayerType<N>::WeightType;
	template <size_t N> using LOutputType = LayerType<N>::OutputType;
	template <size_t N> using LInputType = LayerType<N>::InputType;
	
	static constexpr size_t number_of_layers = sizeof...(Layers);
	
	using InputType = LayerType<0>::InputType;
	using OutputType = LayerType<number_of_layers-1>::OutputType;
	using ScalarType = LayerType<0>::ScalarType;
public :
	FeedbackwardNeuralNetwork(Layers... layers_)
		: network(layers_...)
	{}
	
	/* Feed Forward Algorithm */

private :
	
	template <size_t N>
	OutputType feed_forward(const LInputType<N>& input) {
		static_assert(N < number_of_layers && N >= 0);
		
		std::get<N+1>(outputs) = network.template feed_forward_layer<N>(input);
		
		if constexpr (N == number_of_layers-1)
		{
			return std::get<N+1>(outputs);
		}
		else
		{	return feed_forward<N+1>(std::get<N+1>(outputs));
		}
	}
	
	
	
	template <size_t N>
	void update_weight() {
		get_weight<N>() -= std::get<N>(deltas)*std::get<N>(outputs).transpose();
		if constexpr (N > 0) update_weight<N-1>();
	}
	
public :
	OutputType feed_forward(const InputType& input) {
		return feed_forward<0>(input);
	}
	
	template <size_t N>
	LWeightType<N>& get_weight() { return std::get<N>(network.layers).get_weight(); }
	
	/* Optimization */

private :

public :
	void feed_backward(InputType& input, OutputType& actual, double lambda=1) {
		OutputType out = feed_forward(input);
		std::get<0>(outputs) = input;
		std::get<number_of_layers>(deltas) = derivative<mean_squared_error_loss<OutputType>>::eval(out, actual);
		find_delta<number_of_layers-1>();
		update_weight<number_of_layers-1>();
	}
	
private :
	NeuralNetwork<Layers...> network;
	std::tuple<InputType, typename Layers::OutputType...> outputs;
	std::tuple<typename Layers::OutputType..., OutputType> deltas;
	
};

}

#endif /* SRC_FEED_BACKWARD_HPP_ */
