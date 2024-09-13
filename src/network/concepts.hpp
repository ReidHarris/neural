/*
 * bias_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_CONCEPTS_HPP_
#define SRC_CONCEPTS_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>

namespace neural
{

/* Scalar Concept */
template <typename T> 	constexpr bool 	is_scalar			= false;
template <> 			constexpr bool 	is_scalar<float> 	= true;
template <> 			constexpr bool 	is_scalar<double> 	= true;

template <class T> concept scalar = is_scalar<T>;

/* Matrix Concept */
template <typename T> 							constexpr bool is_matrix 										= false;
template <typename Scalar, int Rows, int Cols>	constexpr bool is_matrix<Eigen::Matrix<Scalar, Rows, Cols>> 	= true;

/* Layer Concept */
template <typename T> 	constexpr bool is_layer = false;

template <class T> concept layer_type = is_layer<T>;

}

#endif /* SRC_CONCEPTS_HPP_ */
