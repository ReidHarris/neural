//
//  player.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//

#ifndef player_hpp
#define player_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <deque>

#include "src/network/math_functions.hpp"
#include "src/network/perceptron_layer.hpp"
#include "src/network/genetic_algorithm_neural_network.hpp"

#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/glu.h>
#elifdef __linux__
	#include <GL/glut.h>
	#include <GL/glu.h>
#endif


#define X_WINDOW_SIZE 400
#define Y_WINDOW_SIZE 400
#define CELL_SIZE 20
#define BOARD_SIZE 20

// Game objects.

struct Cell {
	
	enum Direction {
		left	= 0,
		right	= 1,
		up		= 2,
		down	= 3
	};
	
	Cell() = default;
	Cell(int row_, int col_) : row{row_}, col{col_} {}
	
	void draw() {
		int r = -X_WINDOW_SIZE/2 + row*CELL_SIZE;
		int c = -X_WINDOW_SIZE/2 + col*CELL_SIZE;
        glBegin(GL_POLYGON);
        glVertex2f(c, 				r);
        glVertex2f(c + CELL_SIZE, 	r);
        glVertex2f(c + CELL_SIZE, 	r + CELL_SIZE);
        glVertex2f(c, 				r + CELL_SIZE);
        glEnd();
	}
	
	bool operator==(Cell other) {
		return (row == other.row) && (col == other.col);
	}
	
	Cell move(Cell::Direction dir) {
		switch(dir) {
			case(Direction::left) :
				return Cell(row, col-1);
				break;
			case(Direction::right) :
				return Cell(row, col+1);
				break;
			case(Direction::up) :
				return Cell(row-1, col);
				break;
			case(Direction::down) :
				return Cell(row+1, col);
				break;
			default :
				return Cell(0,0);
		}
	}
	
	bool is_out_of_bounds() {
		return (row < 0 || row >= Y_WINDOW_SIZE || col < 0 || X_WINDOW_SIZE);
	}
	
	int row;
	int col;
};

struct Snake {
	
	Snake() : dir{Cell::Direction::right}
	{
		tail.push_back(Cell());
	}
	
	Cell head() { return tail.front(); }
	
	bool update_position() {
		Cell next = tail.front();
		
		switch (dir) {
			case (Cell::Direction::left) :
				next.col--;
				break;
			case (Cell::Direction::right) :
				next.col++;
				break;
			case (Cell::Direction::up) :
				next.row--;
				break;
			case (Cell::Direction::down) :
				next.row++;
				break;
		}
		
		if (find(tail.begin(), tail.end(), next) != tail.end()) return false;
		if (next.row < 0 || next.row >= BOARD_SIZE) return false;
		if (next.col < 0 || next.col >= BOARD_SIZE) return false;
		
		tail.push_front(next);
		return true;
	}
	
	void delete_tail() { tail.pop_back(); }
	
	
	void draw() {
		glColor3f(0.5f, 0.5f, 0.5f);
		for(auto it = tail.begin(); it != tail.end(); ++it)
			it->draw();
	}
	
	auto begin() { return tail.begin(); }
	auto end() { return tail.end(); }
	
	Cell::Direction dir;
	std::deque<Cell> tail;
};


/*
 *  Game Class
 */
void display();
void gameOver();

struct SnakeGame
{
	static SnakeGame current_game;
	
	SnakeGame() : game_over{false}, score{0}
	{
		snake = Snake();
		food = random_cell();
		game_over_callable = [](){ current_game = SnakeGame(); };
		hunger = 200;
	}
	
	SnakeGame(std::function<void(void)> game_over_) : game_over{false}, score{0}
	{
		snake = Snake();
		food = random_cell();
		game_over_callable = game_over_;
		hunger = 200;
	}
	
	Cell random_cell()
	{
		Cell res = Cell(rand() % BOARD_SIZE, rand() % BOARD_SIZE);
		while (find(snake.tail.begin(), snake.tail.end(), res) != snake.tail.end())
			res = Cell(rand() % BOARD_SIZE, rand() % BOARD_SIZE);
		return res;
	}
	
	void timer(int)
	{
		display();
		game_over = !snake.update_position();
		if(game_over || hunger <= 0) game_over_callable();
		else if(snake.head() != food) {
			snake.delete_tail();
			--hunger;
		}
		else
		{
			score++;
			food = random_cell();
			hunger = 200;
		}
	}
	
	Cell food;
	Snake snake;
	bool game_over;
	int score;
	int hunger;
	std::function<void(void)> game_over_callable;
};



void init(){
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glOrtho(GLdouble(-X_WINDOW_SIZE/2), GLdouble(X_WINDOW_SIZE/2), GLdouble(Y_WINDOW_SIZE/2), GLdouble(-Y_WINDOW_SIZE/2), -1, 1);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    SnakeGame::current_game.food.draw();
    SnakeGame::current_game.snake.draw();
    glFlush();
    glutSwapBuffers();
}

void timer_func(int n) {
	if(n == 0) SnakeGame::current_game.game_over_callable();
	SnakeGame::current_game.timer(0);
	glutTimerFunc(50,timer_func,n-1);
}
   
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: //ESC.
            exit(0);
            break;

        case 'w': //Up.
        	SnakeGame::current_game.snake.dir = Cell::Direction::up;
            break;

        case 's': //Down.
        	SnakeGame::current_game.snake.dir = Cell::Direction::down;
            break;

        case 'a': //Left.
        	SnakeGame::current_game.snake.dir = Cell::Direction::left;
            break;

        case 'd': //Right.
        	SnakeGame::current_game.snake.dir = Cell::Direction::right;
            break;
    }
}

template <typename NetworkType>
struct SnakeGameAI {
	
	SnakeGameAI(NetworkType& nn_) {
		network = nn_;
	}
	
	void set_network(NetworkType& nn_) {
		network = nn_;
	}
	
	NetworkType::InputType input() {
		auto obstacle = [](Cell::Direction dir) {
			Cell c = SnakeGame::current_game.snake.head().move(dir);
			return (c.is_out_of_bounds() || (find(SnakeGame::current_game.snake.begin(), SnakeGame::current_game.snake.end(), c) != SnakeGame::current_game.snake.end()));
		};
		
		return {
			SnakeGame::current_game.snake.head().row/(Y_WINDOW_SIZE/2),
			SnakeGame::current_game.snake.head().col/(X_WINDOW_SIZE/2),
			SnakeGame::current_game.food.row/(Y_WINDOW_SIZE/2),
			SnakeGame::current_game.food.col/(X_WINDOW_SIZE/2),
			obstacle(Cell::Direction::up),
			obstacle(Cell::Direction::down),
			obstacle(Cell::Direction::left),
			obstacle(Cell::Direction::right),
		};
	}
	
	NetworkType::OutputType output() {
		return network.feed_forward(input());
	}
	
	void action() {
		int max_index = 0;
		auto out = output();
		for(int i = 1; i < NetworkType::OutputSize; ++i)
			if(out(max_index) < out(i)) max_index  = i;
		switch(max_index) {
			case(0) :
				SnakeGame::current_game.snake.dir = Cell::Direction::left;
				break;
			case(1) :
				SnakeGame::current_game.snake.dir = Cell::Direction::up;
				break;
			case(2) :
				SnakeGame::current_game.snake.dir = Cell::Direction::down;
				break;
			case(3) :
				SnakeGame::current_game.snake.dir = Cell::Direction::right;
				break;
		}
	}
	
	NetworkType network;
};





template <typename NetworkType>
struct SnakeScorer { 
	using OutputType = int;
	using InputType = NetworkType;
	
	bool compare(OutputType a, OutputType b) {
		return a > b;
	}
	
	OutputType operator()(InputType& input) {
		return 3;
	}
};

template <typename NetworkType>
struct SnakeGeneticAlgorithm {
	template <typename Initializer>
	static void initialize(int n, Initializer& init) {
		genetic_algorithm = neural::GeneticAlgorithm<SnakeScorer<NetworkType>, NetworkType>(n);
		genetic_algorithm.initialize(init);
	}
	
	static void gameOver();
	
	static SnakeGameAI<NetworkType> AI;
	static neural::GeneticAlgorithm<SnakeScorer<NetworkType>, NetworkType> genetic_algorithm;
	static int index;
};

template <typename NetworkType>
neural::GeneticAlgorithm<SnakeScorer<NetworkType>, NetworkType> SnakeGeneticAlgorithm<NetworkType>::genetic_algorithm{200};

template <typename NetworkType>
SnakeGameAI<NetworkType> SnakeGeneticAlgorithm<NetworkType>::AI{SnakeGeneticAlgorithm<NetworkType>::genetic_algorithm.population[0].first};

template <typename NetworkType>
void SnakeGeneticAlgorithm<NetworkType>::gameOver() {
	SnakeGeneticAlgorithm<NetworkType>::genetic_algorithm.population[index].second = SnakeGame::current_game.score;
	if(SnakeGeneticAlgorithm<NetworkType>::index >= SnakeGeneticAlgorithm<NetworkType>::genetic_algorithm.population_size-1) {
		SnakeGeneticAlgorithm<NetworkType>::index = 0;
		SnakeGeneticAlgorithm<NetworkType>::genetic_algorithm.evolve(50, 0.05);
	}
	else ++SnakeGeneticAlgorithm<NetworkType>::index;
	SnakeGeneticAlgorithm<NetworkType>::AI.set_network(SnakeGeneticAlgorithm<NetworkType>::genetic_algorithm.population[SnakeGeneticAlgorithm<NetworkType>::index].first);
	SnakeGame::current_game = SnakeGame(SnakeGeneticAlgorithm<NetworkType>::gameOver);
}

template <typename NetworkType>
int SnakeGeneticAlgorithm<NetworkType>::index = 0;

template <typename NetworkType>
void neural_network_timer(int n) {
	SnakeGeneticAlgorithm<NetworkType>::AI.action();
	SnakeGame::current_game.timer(0);
	glutTimerFunc(1,neural_network_timer<NetworkType>, 0);
}



#endif /* player_hpp */
