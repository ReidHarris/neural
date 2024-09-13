//
//  asteroids_ai.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//

#ifndef ASTEROIDS_AI_HPP_
#define ASTEROIDS_AI_HPP_

#include "asteroids_game.hpp"
#include "src/network/initialization.hpp"
#include "src/network/math_functions.hpp"
#include "src/network/perceptron_layer.hpp"
#include "src/network/recurrent_layer.hpp"
#include "src/network/genetic_algorithm_neural_network.hpp"

namespace asteroids
{

int N_evolve = 50; // How many of the top scorers to use when creating next generation.
static constexpr int number_of_inputs = 16; // Inputs to neural network.
static constexpr int recurrent_inputs = 0;

// Scorer class

template <typename NetworkType>
struct AsteroidScorer {
	static_assert(NetworkType::InputSize == number_of_inputs && NetworkType::OutputSize == 2); 
	using OutputType = int;
	using InputType = NetworkType;
	
	bool compare(OutputType a, OutputType b) {
		return a > b;
	}
	
	OutputType operator()(InputType& input) {
		return 0;
	}
};

// Neural Network Class.

using RL = neural::SigmoidLayer<number_of_inputs+recurrent_inputs,30>;
using SL = neural::TanhLayer<30,2>;
using NN = neural::NeuralNetwork<RL, SL>;

neural::GaussianInitializer gauss(0, 1);
neural::GeneticAlgorithm<AsteroidScorer<NN>, NN> GenAlg(1000);

template <typename NetworkType>
struct AsteroidsGameAI {

	AsteroidsGameAI(NetworkType& nn_) { network = nn_; }
	void set_network(NetworkType& nn_) { network = nn_; }

	NetworkType::OutputType output() {
		return network.feed_forward(AsteroidsGame::current_game.state());
	}

	void action() {
		auto out = output();
		AsteroidsGame::current_game.play.velocity += out(0)*AsteroidsGame::current_game.play.orientation;
		AsteroidsGame::current_game.play.orientation = Eigen::Rotation2D(out(1)/3)*AsteroidsGame::current_game.play.orientation;
	}

	NetworkType network;
};

template <typename NetworkType>
struct AsteroidsGeneticAlgorithm {
	
	template <typename Initializer>
	static void initialize(Initializer& init)
	{
		genetic_algorithm.initialize(init);
	}

	static void gameOver();

	static AsteroidsGameAI<NetworkType> AI;
	static neural::GeneticAlgorithm<AsteroidScorer<NetworkType>, NetworkType> genetic_algorithm;
	static int index;
	static int generation;
	static double rate;
};

template <typename NetworkType>
neural::GeneticAlgorithm<AsteroidScorer<NetworkType>, NetworkType>
AsteroidsGeneticAlgorithm<NetworkType>::genetic_algorithm{5000};

template <typename NetworkType>
double AsteroidsGeneticAlgorithm<NetworkType>::rate {0.05};

template <typename NetworkType>
int AsteroidsGeneticAlgorithm<NetworkType>::index {0};

template <typename NetworkType>
int AsteroidsGeneticAlgorithm<NetworkType>::generation {1};

template <typename NetworkType>
AsteroidsGameAI<NetworkType> AsteroidsGeneticAlgorithm<NetworkType>::AI {AsteroidsGeneticAlgorithm<NetworkType>::genetic_algorithm.population[0].first};

template <typename NetworkType>
void AsteroidsGeneticAlgorithm<NetworkType>::gameOver() {
	std::cout << "\r\t\t\t\r"
		<< AsteroidsGeneticAlgorithm<NetworkType>::index 
		<< " : " 
		<< AsteroidsGame::current_game.max_score
		<< std::flush;
	AsteroidsGeneticAlgorithm<NetworkType>::genetic_algorithm.population[index].second = AsteroidsGame::current_game.score;
	AsteroidsGeneticAlgorithm<NetworkType>::AI.network = AsteroidsGeneticAlgorithm<NetworkType>::genetic_algorithm.population[index].first;
	AsteroidsGame::current_game.reset();
	AsteroidsGeneticAlgorithm<NetworkType>::index++;
	if(AsteroidsGeneticAlgorithm<NetworkType>::index == AsteroidsGeneticAlgorithm<NetworkType>::genetic_algorithm.population.size()) {
		AsteroidsGeneticAlgorithm<NetworkType>::genetic_algorithm.evolve(N_evolve, AsteroidsGeneticAlgorithm<NetworkType>::rate);
		AsteroidsGeneticAlgorithm<NetworkType>::index = 0;
		AsteroidsGeneticAlgorithm<NetworkType>::generation++;
		std::cout << "\n" << AsteroidsGeneticAlgorithm<NetworkType>::generation << std::endl;
		AsteroidsGame::current_game.max_score = 0;
	}
	AsteroidsGame::current_game.gen = std::default_random_engine(AsteroidsGeneticAlgorithm<NetworkType>::generation);
	AsteroidsGame::current_game.addInitialParticle();
}





using RL = neural::SigmoidLayer<16,30>;
using SL = neural::TanhLayer<30,2>;
using NN = neural::NeuralNetwork<RL, SL>;

}
#endif /* ASTEROIDS_AI_HPP_ */
