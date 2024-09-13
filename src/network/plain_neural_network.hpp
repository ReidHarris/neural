/*
 * bias_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_LAYERS_PLAIN_NEURAL_NETWORK_HPP_
#define SRC_LAYERS_PLAIN_NEURAL_NETWORK_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>
#include <string>
#include <fstream>

#include "perceptron_layer.hpp"

namespace neural
{

template <typename... Layers>
struct LayerInfo {
	template <size_t N> using LayerType   = std::tuple_element<N, std::tuple<Layers...>>::type;
	template <size_t N> using LWeightType = LayerType<N>::WeightType;
	template <size_t N> using LOutputType = LayerType<N>::OutputType;
	template <size_t N> using LInputType  = LayerType<N>::InputType;
};

template<typename... Layers>
constexpr bool check_input_output_layer = false;

template<typename L>
constexpr bool check_input_output_layer<L> = true;

template<typename L1, typename L2, typename... Layers>
constexpr bool check_input_output_layer<L1, L2, Layers...> 
	= check_input_output_layer<L2, Layers...> && std::is_same_v<typename L1::OutputType, typename L2::InputType>;

struct FeedbackwardPolicy {
	struct None;
	struct SGD;
};

template <typename P>
constexpr bool is_feedback_policy = false;

template <>
constexpr bool is_feedback_policy<FeedbackwardPolicy::None> = true;

template <>
constexpr bool is_feedback_policy<FeedbackwardPolicy::SGD> = true;

template <typename P>
concept feedback_policy = is_feedback_policy<P>;

template <typename FirstLayer, typename... Layers>
struct IntermediateResults
{
	using TupleType = std::tuple<typename FirstLayer::InputType, typename FirstLayer::OutputType, typename Layers::OutputType...>;
	
	template <size_t N> 
	using Type = std::tuple_element<N, TupleType>::type;
	
	template <int N> 
	void set(Type<N>& result) 
	{
		std::get<N>(intermediate_results) = result;
	}
	
	template <int N>
	const Type<N>& get() const
	{
		return std::get<N>(intermediate_results);
	}
	
	TupleType intermediate_results;
};

template<feedback_policy Policy, typename... Layers>
struct Optimizer;

template <typename... Layers>
struct Optimizer<FeedbackwardPolicy::None, Layers...> {
	
};
/*
template <typename... Layers>
struct Optimizer<FeedbackwardPolicy::SGD, Layers...> {
	using SavedResults = IntermediateResults<Layers...>;
	
	template <size_t N>
	void find_delta() {
		static_assert(N >= 0 && N < sizeof...(Layers));
		
		LInputType<N> x_0 = std::get<N>(outputs);
		LWeightType<N> W_0 = get_weight<N>();
		
		constexpr if (N < sizeof...(Layers) - 1)
		{
			LOutputType<N+1> 	x_1 = std::get<N+1>	(deltas);
			LWeightType<N+1>	W_1 = get_weight<N+1>();
			
			std::get<N>(deltas) = ( (W_0*x_0).unaryExpr(&derivative<typename LayerType<N>::Activation>::eval) )
					.cwiseProduct(W_1.transpose()*x_1);
		}
		else if (N == sizeof...(Layers) - 1)
		{
			std::get<N>(deltas) = ( std::get<N+1>(deltas) )
				.cwiseProduct((W_0*x_0).unaryExpr(&derivative<typename LayerType<N>::Activation>::eval));
		}
		else if (N == sizeof...(Layers))
		{
			std::get<N>(deltas) = derivative<mean_squared_error_loss<OutputType>>::eval(get<N>(), actual);
		}
		
		if constexpr (N > 0) find_delta<N-1>();
	}
	
	
	template <int N>
	void set(typename SavedResults::Type<N>& result)
	{
		intermediate_results.set<N>(result);
	}
	
	template <int N>
	typename SavedResults::Type<N>& get() const
	{
		return intermediate_results.get<N>();
	}
	
	IntermediateResults<Layers...> intermediate_results;
	
}; */

template </* feedback_policy GradientDescentPolicy = FeedbackwardPolicy::None, */ typename... Layers>
class NeuralNetwork
{
public :
	template <size_t N> using LayerType   = std::tuple_element<N, std::tuple<Layers...>>::type;
	template <size_t N> using LWeightType = LayerType<N>::WeightType;
	template <size_t N> using LOutputType = LayerType<N>::OutputType;
	template <size_t N> using LInputType  = LayerType<N>::InputType;
	
	static constexpr size_t number_of_layers = sizeof...(Layers);
	
	using InputType  = LayerType<0>::InputType;
	using OutputType = LayerType<number_of_layers-1>::OutputType;
	using ScalarType = LayerType<0>::ScalarType;
	
	static constexpr int InputSize  = LayerType<0>::InputSize;
	static constexpr int OutputSize = LayerType<number_of_layers-1>::OutputSize;
	
	//using TrainingPolicy = GradientDescentPolicy;
	//static constexpr bool UsesGradientDescent = !std::is_same_v<FeedbackwardPolicy::None, TrainingPolicy>;
public :
	NeuralNetwork() {}
	
	NeuralNetwork(Layers... layers_)
	{
		static_assert(check_input_output_layer<Layers...>);
		layers = std::make_tuple(layers_...);
		
	}
	
	/* Feed Forward Algorithm */

private :
	
	template <size_t N>
	OutputType feed_forward_to_final(const LInputType<N>& input)
	{
		static_assert(N < number_of_layers && N >= 0);
		
		LOutputType<N> res = std::get<N>(layers).feed_forward(input);
		if constexpr (N == number_of_layers-1) return res;
		else return feed_forward_to_final<N+1>(res);
	}
	
public :
	OutputType feed_forward(const InputType& input)
	{
		return feed_forward_to_final<0>(input);
	}
	
	
	
	void operator=(NeuralNetwork<Layers...> other)
	{
		layers = other.layers;
	}
	
	template <typename Initializer>
	void initialize(Initializer& init)
	{
		std::apply([&init](Layers&... l)
					{
						((l.initialize(init)), ...);
					}, layers);
	}
	
	std::tuple<Layers...> layers;
	// Optimizer<TrainingPolicy> optimizer;
};

// Reading and writing functions.

template <typename...Layers>
void save_to_file(std::ofstream& file, const NeuralNetwork<Layers...>& nn)
{
	if(!file.is_open())
	{
		std::cout << "File not open. Cannot be saved." << std::endl;
		return;
	}
	std::apply( [&file](const Layers&... layers)
				{
					((save_to_file(file, layers)), ...);
				}, nn.layers);
}

template <typename... Layers>
void read_from_file(std::ifstream& file, NeuralNetwork<Layers...>& nn)
{
	if(! file.is_open())
	{
		std::cout << "File not open. Cannot be read." << std::endl;
		return;
	}
	std::apply(
		[&file](Layers&...layers) 
		{
			((read_from_file(file, layers)), ...);
		}, nn.layers);
		
}

}


#endif /* SRC_LAYERS_PLAIN_NEURAL_NETWORK_HPP_ */
