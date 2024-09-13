//
//  player.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//

#ifndef SNAKE_HPP_
#define SNAKE_HPP_

#include "snake.hpp"


using RL = neural::SigmoidLayer<8, 20>;
using SL = neural::SigmoidLayer<20, 4>;
using NN = neural::NeuralNetwork<RL, SL>;

SnakeGame SnakeGame::current_game = SnakeGame(SnakeGeneticAlgorithm<NN>::gameOver);

int main(int argc, char **argv) {
	
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(X_WINDOW_SIZE, Y_WINDOW_SIZE);
    glutCreateWindow("Snake");   
    init();

    glutDisplayFunc(display);
    
    using GA = neural::GeneticAlgorithm<SnakeScorer<NN>, NN>;
    using SGA = SnakeGeneticAlgorithm<NN>;
    neural::GaussianInitializer gauss(0,1);
    SGA::initialize<neural::GaussianInitializer>(200, gauss);
    
    
    
    neural_network_timer<NN>(0);
	
    //glutKeyboardFunc(keyboard);
	
	
	glutMainLoop();
}

#endif /* SNAKE_HPP_ */
