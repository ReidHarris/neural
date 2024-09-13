/*
 * perceptron_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_PERCEPTRON_LAYER_HPP_
#define SRC_PERCEPTRON_LAYER_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>
#include <type_traits>
#include <fstream>
#include <iostream>
#include <string>
#include "initialization.hpp"
#include "concepts.hpp"

namespace neural
{

/* Perceptron Layer */

template <typename ScalarType = double>
class Layer {
	virtual Layer(int inputs, int outputs) = 0;
	virtual Vector feed_forward(Vector input) = 0;
	virtual Matrix get_weight() = 0;
	virtual void set_weight() = 0;
	
};

template < typename ScalarType>
class PerceptronLayerDynamic : public Layer
{
public :
	static_assert(NumInputs > 0 && NumOutputs > 0);
	using Scalar = ScalarType;
	
	static constexpr bool HasParameters = true;
	
	using InputType  = Eigen::Matrix<Scalar, Dynamic, 1>;
	using OutputType = Eigen::Matrix<Scalar, Dynamic, 1>;
	using WeightType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
public :
	PerceptronLayerDynamic(int inputs, int outputs) : weight(outputs, inputs) {}
	
	template <typename Initializer>
	void initialize(Initializer& init) { init.initialize(*this); }
	
	OutputType feed_forward(const InputType& input) { return (weight*input).unaryExpr(&activation_function); }
	
	void set_weight(WeightType& weight_) { 
		assert(weight_.rows() == outputs && weight_.cols() == inputs);
		weight = weight_;
	}
	const WeightType& get_weight() const { return weight; }
	void set_weight_entry(ScalarType x, int i, int j) { weight(i,j) = x; }
	
	int inputs() const { return inputs; }
	int outputs() const { return outputs; }
	
private :
	WeightType weight;
	std::function<Scalar(Scalar)> activation_function;
	int inputs;
	int outputs;
};

template <typename ScalarType>
class NeuralNetwork {
	
};

template <typename Scalar>
void read_from_file(std::ifstream& file, PerceptronLayerDynamic<Scalar>& layer)
{
	if(!file.is_open()) return;
	std::string matrix_string;
	getline(file, matrix_string);
	std::stringstream matrix_stream(matrix_string);
	std::string entry;
	int number_of_parameters = layer.inputs()*layer.outputs();
	
	int i=0, r=0, c=0;
	
	
	while(std::getline(matrix_stream, entry, ','))
	{
		assert(i < number_of_parameters);
		layer.set_weight_entry(std::stod(entry),r,c);
		r = r + (c+1)/NumInputs;
		c = (c+1)%NumInputs;
		++i;
	}
	assert(i == number_of_parameters);
}

template <typename Scalar, int NumInputs, int NumOutputs, template<typename> class Activation>
void save_to_file(std::ofstream& file, const PerceptronLayer<Scalar, NumInputs, NumOutputs, Activation>& layer)
{
	Eigen::IOFormat CommaInitFmt(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
	if(file.is_open()) file << layer.get_weight().format(CommaInitFmt) << std::endl;
}

/* Specializations */

template <int NumInputs, int NumOutputs>
using SigmoidLayer = PerceptronLayer<double, NumInputs, NumOutputs, sigmoid>;

template <int NumInputs, int NumOutputs>
using TanhLayer = PerceptronLayer<double, NumInputs, NumOutputs, tanh>;

template <int NumInputs, int NumOutputs>
using ReLULayer = PerceptronLayer<double, NumInputs, NumOutputs, ReLU>;

/* Type Traits */

template <typename Scalar, int NumInputs, int NumOutputs, template<typename> class ActivationFunction>
constexpr bool is_layer<PerceptronLayer<Scalar, NumInputs, NumOutputs, ActivationFunction>> 	= true;

template <int NumInputs, int NumOutputs>
constexpr bool is_layer<SigmoidLayer<NumInputs, NumOutputs>	> 	= true;

template <int NumInputs, int NumOutputs>
constexpr bool is_layer<ReLULayer<NumInputs, NumOutputs>	> 	= true;

template <int NumInputs, int NumOutputs>
constexpr bool is_layer<TanhLayer<NumInputs, NumOutputs>	> 	= true;

}

#endif /* SRC_PERCEPTRON_LAYER_HPP_ */
