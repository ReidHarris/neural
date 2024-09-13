/*
 * bias_layer.hpp
 *
 *  Created on: Dec 4, 2023
 *      Author: reidharris
 */

#ifndef SRC_GENETIC_ALGORITHM_HPP_
#define SRC_GENETIC_ALGORITHM_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cassert>

#include <random>

#include "plain_neural_network.hpp"



namespace neural
{

template <typename ScorerT, typename NN>
class GeneticAlgorithm
{
public :
	template <size_t N> using LayerType   = NN::template LayerType<N>;
	template <size_t N> using LWeightType = LayerType<N>::WeightType;
	template <size_t N> using LOutputType = LayerType<N>::OutputType;
	template <size_t N> using LInputType  = LayerType<N>::InputType;
	
	static constexpr size_t number_of_layers = NN::number_of_layers;
	
	using InputType  = LayerType<0>::InputType;
	using OutputType = LayerType<number_of_layers-1>::OutputType;
	using ScalarType = LayerType<0>::ScalarType;
	
	using NetworkType    = NN;
	using PopulationType = std::pair < NetworkType, typename ScorerT::OutputType>;
	
public :
	GeneticAlgorithm(unsigned int population_size_)
		: population(population_size_)
		, population_size{population_size_}
	{
		static_assert(std::is_same_v<typename ScorerT::InputType, NetworkType>);
	}
	
	/* Feed Forward Algorithm */

public :
	template <typename Initializer>
	void initialize(Initializer& init)
	{
		for (auto& nn : population) nn.first.initialize(init);
	}
	
	/* Crossover Functions */
	
	template <int N>
	void get_child_of_layer(const LayerType<N>& layer1, const LayerType<N>& layer2, LayerType<N>& child1, LayerType<N>& child2)
	{
		std::uniform_int_distribution dist(0,1);
		
		for(int i = 0; i < LayerType<N>::OutputSize; ++i)
		{
			for(int j = 0; j < LayerType<N>::InputSize; ++j)
			{
				if(dist(gen))
				{
					child1.get_weight()(i,j) = layer1.get_weight()(i,j);
					child2.get_weight()(i,j) = layer2.get_weight()(i,j);
				}
				else
				{
					child1.get_weight()(i,j) = layer2.get_weight()(i,j);
					child2.get_weight()(i,j) = layer1.get_weight()(i,j);
				}
			}
		}
	}
	
	template <int N>
	void get_child(const NetworkType& nn1, const NetworkType& nn2, NetworkType& nn_child1, NetworkType& nn_child2)
	{
		static_assert(N >= 0 && N < number_of_layers);
		get_child_of_layer<N>(std::get<N>(nn1.layers), std::get<N>(nn2.layers), std::get<N>(nn_child1.layers), std::get<N>(nn_child2.layers));
		
		if constexpr (N < number_of_layers-1) get_child<N+1>(nn1, nn2, nn_child1, nn_child2);
	}
	
	void get_child(const NetworkType& nn1, const NetworkType& nn2, NetworkType& nn_child1, NetworkType& nn_child2)
	{
		get_child<0>(nn1, nn2, nn_child1, nn_child2);
	}
	
	void mutate(NetworkType& nn, double rate = 0.05)
	{
		std::normal_distribution<double> dist(0,rate);
		
		auto mutate_function = [&dist, this]<typename Layer>(Layer& layer){
			for (int i = 0; i < Layer::OutputSize; ++i)
				for (int j = 0; j < Layer::InputSize; ++j)
					layer.get_weight()(i,j) += dist(this->gen);
		};
		
		std::apply([&mutate_function]<typename...L>(L&... layers)
				{
					((mutate_function(layers)),...);
				}, nn.layers);
	}
	
public :
	
	void get_scores()
	{
		for(int i = 0; i < population.size(); i++)
		{
			population[i].second = scorer(population[i].first);
		}
	}
	
	void sort_by_scores() {
		std::sort(population.begin(), population.end(), 
			[=](PopulationType& a, PopulationType& b)
			{
				return scorer.compare(a.second, b.second);
			});
	}
	
	void evolve(int number_of_parents, double rate) 
	{
		assert(number_of_parents >= 2 && number_of_parents < population_size);
		sort_by_scores();
		std::ofstream file;
		file.open("scores.txt", std::ios::out | std::ios_base::app);
		if(file.is_open())
		{
			for(int i = 0; i < population.size(); ++i) {
				file << population[i].second;
				if(i == population.size() - 1) file << ";" << std::endl;
				else file << ", ";
			}
		} 
		file.close();
		std::vector<PopulationType> next_generation (population_size);
		for(int i = 0; i < number_of_parents; ++i) next_generation[i] = population[i];
		std::vector<int> vec(population_size);
		for(int i = 0; i < population_size; ++i)
		{
			vec.push_back(population[i].second);
		}
		std::discrete_distribution<int> dist(vec.begin(), vec.end());
		
		for(int i = number_of_parents; i < population_size; ++i)
		{
			int index1 = dist(gen) % population_size;
			int index2 = dist(gen) % population_size;
			
			while(index1 == index2) {
				index1 = dist(gen) % population_size;
				index2 = dist(gen) % population_size;
			}
			
			const NetworkType& parent1 = population[index1].first;
			const NetworkType& parent2 = population[index2].first;
			
			NetworkType child1, child2;
			
			get_child(parent1, parent2, child1, child2);
			
			mutate(child1, rate);
			mutate(child2, rate);
			
			next_generation[i] = {child1, 0};
			if (i >= population_size-1) break;
			next_generation[i+1] = {child2, 0};
		}
		population = next_generation;
	}
	
public :
	std::vector<PopulationType> population;
	ScorerT scorer;
	size_t population_size;
	std::default_random_engine gen;
};

template <typename ScoreT, typename...Layers>
void save_to_file(std::ofstream& file, const GeneticAlgorithm<ScoreT, Layers...>& genetic_algorithm)
{
	if(!file.is_open()) return;
	for (auto p : genetic_algorithm.population)
		save_to_file(file, p.first);
}

template <typename ScoreT, typename...Layers>
void read_from_file(std::ifstream& file, GeneticAlgorithm<ScoreT, Layers...>& genetic_algorithm)
{
	if(!file.is_open()) return;
	for(auto& p : genetic_algorithm.population)
		read_from_file(file, p.first);
}

}

#endif /* SRC_FEED_BACKWARD_HPP_ */
