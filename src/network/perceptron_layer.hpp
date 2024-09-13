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

template < typename Scalar, int NumInputs, int NumOutputs, template<typename> class ActivationFunction>
class PerceptronLayer
{
public :
	static_assert(NumInputs > 0 && NumOutputs > 0);
	using ScalarType = Scalar;
	static constexpr int InputSize = NumInputs;
	static constexpr int OutputSize = NumOutputs;
	
	static constexpr bool HasParameters = true;
	
	using InputType  = Eigen::Matrix<ScalarType, NumInputs, 1>;
	using OutputType = Eigen::Matrix<ScalarType, NumOutputs, 1>;
	using WeightType = Eigen::Matrix<ScalarType, NumOutputs, NumInputs>;
	using Activation = ActivationFunction<ScalarType>;
public :
	PerceptronLayer() {}
	PerceptronLayer(WeightType w_) : weight(w_) {};
	
	template <typename Initializer>
	void initialize(Initializer& init) { init.initialize(*this); }
	
	inline OutputType feed_forward(const InputType& input) { return (weight*input).unaryExpr(&Activation::eval); }
	
	inline WeightType& get_weight() { return weight; }
	
	inline const WeightType& get_weight() const { return weight; }
	
private :
	WeightType weight;
};

template <typename Scalar, int NumInputs, int NumOutputs, template<typename> class Activation>
void read_from_file(std::ifstream& file, PerceptronLayer<Scalar, NumInputs, NumOutputs, Activation>& layer)
{
	if(!file.is_open()) return;
	std::string matrix_string;
	getline(file, matrix_string);
	std::stringstream matrix_stream(matrix_string);
	std::string entry;
	int number_of_parameters = NumInputs*NumOutputs;
	
	int i=0, r=0, c=0;
	
	while(std::getline(matrix_stream, entry, ','))
	{
		assert(i < number_of_parameters);
		layer.get_weight()(r,c) = std::stod(entry);
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
