/*
 * linear_model.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_LINEAR_MODEL_HPP_
#define SRC_LINEAR_MODEL_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>

namespace neural
{
	
template <typename ScalarType>
struct LearningRate {
	LearningRate(ScalarType r_) : rate{r_} {}
	virtual void update() = 0;
	ScalarType get_rate() { return rate; }
private :
	ScalarType rate;
};

template <typename ScalarType>
struct ConstantLearningRate : public LearningRate<ScalarType> {
	void update() override {};
};

template <typename ScalarType>
struct ExpLearningRate : public LearningRate<ScalarType> {
	ExpLearningRate(ScalarType r_, ScalarType d_)
		: LearningRate<ScalarType>(r_),
		  decay{d_} {}

	void update() {
		rate *= decay;
	}
	
	ScalarType decay;
};

template <typename ScalarType, unsigned int NumInputs, unsigned int NumOutputs>
struct LinearModel {
	using ParameterType = Eigen::Matrix<ScalarType, NumOutputs, NumInputs>;
	using InputType = Eigen::Matrix<ScalarType, NumInputs, 1>;
	using OutputType = Eigen::Matrix<ScalarType, NumOutputs, 1>;
	
	LinearModel() 
		: parameter{ParameterType::Zero()},
		  bias{OutputType::Zero()} {}
	
	LinearModel(ParameterType parameter_, OutputType bias_)
		: parameter{parameter_},
		  bias{bias_} {}
	
	OutputType predict(InputType& input) { return parameter * input + bias; }
	ParameterType& get_param() { return parameter; }
	OutputType& get_bias() { return bias; }
	virtual void train(InputType&, OutputType&) = 0;

private :
	ParameterType parameter;
	OutputType bias;
};

template <typename ScalarType, unsigned int NumInputs, unsigned int NumOutputs>
struct GradientDescentLinearModel : public LinearModel<ScalarType, NumInputs, NumOutputs> {
	GradientDescentLinearModel()
		: LinearModel<ScalarType, NumInputs, NumOutputs>(), learning_rate{ConstantLearningRate(1)} {}
	
	GradientDescentLinearModel(LearningRate learn_)
		: learning_rate {learn_} {}
	
	void train(InputType& input, OutputType& actual) override
	{
		auto error = predict(input) - actual;
		get_param() -= learning_rate.get_rate() * error * input.transpose();
		get_bias() -= learning_rate.get_rate() * error;
		learning_rate.update();
	}
	
	LearningRate learning_rate;
};

template <typename ModelType>
void gradient_descent(double lambda, ModelType& model, typename ModelType::InputType& input, typename ModelType::OutputType& actual) {
	auto error = model.predict(input) - actual;
	model.get_param() -= lambda * error * input.transpose();
	model.get_bias() -= lambda * error;
}


}

#endif /* SRC_LINEAR_MODEL_HPP_ */
