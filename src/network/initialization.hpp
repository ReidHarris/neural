/*
 * bias_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_INITIALIZATION_HPP_
#define SRC_INITIALIZATION_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>

#include <random>

#include "concepts.hpp"

namespace neural
{


class GaussianInitializer {
public:
	GaussianInitializer(double mean_, double stddev_)
		: distribution{mean_, stddev_}
	{}
	
	template<typename Scalar, int Rows, int Cols>
	void initialize(Eigen::Matrix<Scalar, Rows, Cols>& mat) {
		for (int i = 0; i < Rows; ++i) {
			for (int j = 0; j < Cols; ++j) {
				mat(i,j) = distribution(generator);
			}
		}
	}
	
	template<layer_type LayerT>
	void initialize(LayerT& layer) {
		for (int i = 0; i < LayerT::OutputSize; ++i) {
			for (int j = 0; j < LayerT::InputSize; ++j) {
				layer.get_weight()(i,j) = distribution(generator);
			}
		}
	}
	
private :
	std::default_random_engine generator;
	std::normal_distribution<double> distribution;
};

class UniformInitializer {
public:
	UniformInitializer(double left_, double right_)
		: distribution{left_, right_}
	{}
	
	template<typename LayerT>
	void initialize(LayerT& layer) {
		for (int i = 0; i < LayerT::OutputSize; ++i) {
			for (int j = 0; j < LayerT::InputSize; ++j) {
				layer.get_weight()(i,j) = distribution(generator);
			}
		}
	}
	
private :
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;
};

class XavierInitializer {
public :
	XavierInitializer()
		: distribution{-1, 1}
	{}
	
	template<typename LayerT>
	void initialize(LayerT& layer) {
		for (int i = 0; i < LayerT::OutputSize; ++i) {
			for (int j = 0; j < LayerT::InputSize; ++j) {
				layer.get_weight()(i,j) = sqrt((LayerT::Outputs+LayerT::Inputs)/6.0) * distribution(generator);
			}
		}
	}
	
private :
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;
};

}

#endif /* SRC_INITIALIZATION_HPP_ */
