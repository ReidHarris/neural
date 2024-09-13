//
//  player.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//
#ifndef ASTEROIDS_HPP_
#define ASTEROIDS_HPP_

#include "asteroids_game.hpp"
#include "asteroids_ai.hpp"
#include "asteroids_game_func.hpp"

#define GL_SILENCE_DEPRECATION


using namespace asteroids; 
AsteroidsGame AsteroidsGame::current_game = AsteroidsGame(AsteroidsGeneticAlgorithm<NN>::gameOver);


int main(int argc, char **argv) {
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(X_WINDOW_SIZE, Y_WINDOW_SIZE);
    glutCreateWindow("Asteroid AI");
    init();

	neural::GaussianInitializer gauss(0, 1);
	
	if (argc > 1) {
		std::ifstream file(argv[1], std::ios::out | std::ios::in);
		
		if(file.is_open())
		{
			std::string generation;
			getline(file, generation); 
			AsteroidsGeneticAlgorithm<NN>::generation = std::stoi(generation);
			AsteroidsGame::current_game.gen = std::default_random_engine(AsteroidsGeneticAlgorithm<NN>::generation);
			read_from_file(file, AsteroidsGeneticAlgorithm<NN>::genetic_algorithm);
			file.close();
		}
	}
	else  AsteroidsGeneticAlgorithm<NN>::initialize<neural::GaussianInitializer>(gauss);
	
	std::cout << "\n" << AsteroidsGeneticAlgorithm<NN>::generation << std::endl;
	
	if (argc > 2) {
		displayOn = std::stoi(argv[2]);
	}
	
	AsteroidsGame::current_game.addInitialParticle();
    glutDisplayFunc(display);
    timer_func<NN>(30);
    //add_particle_func(50); 
	
	   
    glutKeyboardFunc(keyboard);
	
	glutMainLoop();
}

#endif /* ASTEROIDS_HPP_ */
