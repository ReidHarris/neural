//
//  asteroids_ai.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//

#include "asteroids_ai.hpp"

#ifndef ASTEROIDS_GAME_FUNC_HPP_
#define ASTEROIDS_GAME_FUNC_HPP_

namespace asteroids
{
bool loadFromFile = true;
bool displayOn = true;

void init() {
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glOrtho(GLdouble(-X_WINDOW_SIZE/2), GLdouble(X_WINDOW_SIZE/2), GLdouble(Y_WINDOW_SIZE/2), GLdouble(-Y_WINDOW_SIZE/2), -1, 1);
}

void drawBitmapText(std::string s, float x, float y)
{
	glRasterPos2f(x, y);
	for(char c : s)  glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, c);
}

void drawLineFromShip(double distance, double theta) {
	Eigen::Vector2d e1 {1,0};
	double orientation_angle = angle(AsteroidsGame::current_game.play.orientation, e1);
	double x = AsteroidsGame::current_game.play.position(0) + distance*cos(theta - orientation_angle);
	double y = AsteroidsGame::current_game.play.position(1) + distance*sin(theta - orientation_angle);
	
	glBegin(GL_LINES);
	glVertex2f(AsteroidsGame::current_game.play.position(0), AsteroidsGame::current_game.play.position(1));
	glVertex2f(x, y);
	glEnd();
	
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glColor3f(1,1,1);
	
	// Display stats.
	drawBitmapText("Maximum Score: " 		+ std::to_string(AsteroidsGame::current_game.number_of_particles), 	-X_WINDOW_SIZE/2+50, Y_WINDOW_SIZE/2-94);
	drawBitmapText("Score: " 				+ std::to_string(AsteroidsGame::current_game.score), 	-X_WINDOW_SIZE/2+50, Y_WINDOW_SIZE/2-70);
	drawBitmapText("Agent Number: " 		+ std::to_string(AsteroidsGeneticAlgorithm<NN>::index), 		-X_WINDOW_SIZE/2+50, Y_WINDOW_SIZE/2-46);
	drawBitmapText("Generation: " 			+ std::to_string(AsteroidsGeneticAlgorithm<NN>::generation), 	-X_WINDOW_SIZE/2+50, Y_WINDOW_SIZE/2-22);
	
	for (int i = 0; i < AsteroidsGame::current_game.number_of_particles; ++i) 
		AsteroidsGame::current_game.particles[i].draw(); //Show particles.
	AsteroidsGame::current_game.play.draw(); //Show player.
	
	
	auto res = AsteroidsGame::current_game.state();
	Eigen::Vector2d e1 {1,0};
	
	glColor3f(1.0f, 0, 0);
	for(int i = 0; i < 8; ++i) {
		drawLineFromShip(-res(i), -PI + (double)i*PI/4);
	}
	glColor3f(0,0,1);
	drawLineFromShip(100, 0);
	
	
	glFlush();
	glutSwapBuffers();
}

bool escape = false;

template <typename NetworkType>
void timer_func(int n)
{
	if(displayOn) display();
	if(n==0) 
	{
		AsteroidsGame::current_game.addRandomParticle(50);
		n = 30;
	}
	if(escape) return;
	AsteroidsGeneticAlgorithm<NetworkType>::AI.action();
	if(AsteroidsGame::current_game.game_over)
	{
		AsteroidsGame::current_game.game_over_callable();
		glutTimerFunc(3, timer_func<NetworkType>, 30);
	}
	else
	{
		AsteroidsGame::current_game.update();
		glutTimerFunc(0,timer_func<NetworkType>, n-1);
	}
}

void add_particle_func(int size)
{  
	if(escape) return;
	AsteroidsGame::current_game.addRandomParticle(size); 
	glutTimerFunc(300, add_particle_func, size);
}  
   
void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 27: //ESC.
            std::ofstream file("parameters-v1.txt", std::ios::out | std::ios::trunc);
			if(file.is_open())
			{
				file << AsteroidsGeneticAlgorithm<NN>::generation << std::endl;   
				save_to_file(file, AsteroidsGeneticAlgorithm<NN>::genetic_algorithm);
			} 
			file.close(); 
			escape = true;
			exit(0);
            break;
	}
}
	
	
	
} 

#endif /* ASTEROIDS_GAME_FUNC_HPP_ */
