/*
 * bias_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_ACTIVATIONS_HPP_
#define SRC_ACTIVATIONS_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>

namespace neural
{
	
	
	/* Activation Functions. */
	
template <typename T>
struct sigmoid {
	static T eval(T x) { return 1.0 / (1.0 + exp(-x)); }
};

template <typename T>
struct tanh {
	static T eval(T x) { return std::tanh(x); }
};

template <typename T>
struct ReLU {
	static T eval(T x) { return (x>0) ? x : 0; }
};

template <typename T>
struct mean_squared_error_loss {
	static T::Scalar eval(T x, T y) { return (x-y).norm(); }
};

	/* Derivatives */

template <typename F>
struct derivative;

template <typename T>
struct derivative<sigmoid<T>> {
	static T eval(T x) { T sig_ = sigmoid<T>::eval(x); return sig_*(1-sig_); }
};

template <typename T>
struct derivative<tanh<T>> {
	static T eval(T x) { return 1-pow(tanh<T>::eval(x), 2); }
};

template <typename T>
struct derivative<ReLU<T>> {
	static T eval(T x) { return (x>0) ? 1 : 0; }
};

template <typename T>
struct derivative<mean_squared_error_loss<T>> {
	static T eval(T x, T y) { return 2*(x-y); }
};





}

#endif /* SRC_ACTIVATIONS_HPP_ */
