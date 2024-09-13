//
//  asteroids_game.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//

#ifndef ASTEROIDS_GAME_HPP_
#define ASTEROIDS_GAME_HPP_

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include "math.hpp"

#define GL_SILENCE_DEPRECATION

#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/glu.h>
#elifdef __linux__
	#include <GL/glut.h>
	#include <GL/glu.h>
#endif



#define MAX_VELOCITY 3

namespace asteroids
{
	
void display();


/*
 *  Game Class
 */
 
struct AsteroidsGame
{
	
	Eigen::Rotation2D<double> leftrot {PI/20};
	Eigen::Rotation2D<double> rightrot {-PI/20};
	
	
	struct Object{
		Eigen::Vector2d position, velocity;
		double radius;
		Object()
			: position{0,0}
			, velocity{0,0}
			, radius{0} {};
		
		virtual ~Object() = default;

		virtual void updatePosition(){ position += velocity; }
		virtual bool checkBounds() = 0;
		virtual void draw() = 0;
	};

	
	double distance(Object& p1, Object& p2){
		return (p1.position - p2.position).norm();
	}

	struct Player : public Object{
		Eigen::Vector2d orientation;
		bool up, down, left, right;

		Player()
			: orientation 	({0, 1})
			, up 			(false)
			, down 			(false)
			, left 			(false)
			, right 		(false)
		{
			position << 0, 0;
			velocity << 0, 0;
		}

		// Return if player is out of bounds. If yes, update position.
		bool checkBounds() override {
			return (abs(position(0)) >= X_WINDOW_SIZE/2) || (abs(position(1)) >= Y_WINDOW_SIZE/2);
			if(abs(position(0)) >= X_WINDOW_SIZE/2) { position(0) *= -0.99; return true; }
			if(abs(position(1)) >= Y_WINDOW_SIZE/2) { position(1) *= -0.99; return true; }
			return false;
		}
		
		void updatePosition() override { 
			position += velocity; 
			velocity *= 0.85;
		}
		
		void draw() override {
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_POLYGON);
			glVertex2f(position(0) - 5*orientation(0) - 7*orientation(1), position(1) - 5*orientation(1) + 7*orientation(0));
			glVertex2f(position(0) - 5*orientation(0) + 7*orientation(1), position(1) - 5*orientation(1) - 7*orientation(0));
			glVertex2f(position(0) - 5*orientation(0) + 20*orientation(0), position(1) - 5*orientation(1) + 20*orientation(1));
			glEnd();
		}
		
	};

	struct Particle : public Object {
	public:
		Particle() = default;

		Particle(Eigen::Vector2d pos_, Eigen::Vector2d vel_, double r)
		{
			position = pos_;
			velocity = vel_;
			radius = r;
		}

		// Return if particle is out of bounds.
		bool checkBounds() override
		{
			return position(0) > X_WINDOW_SIZE/2 + 100 || position(1) > Y_WINDOW_SIZE/2 + 100 || position(0) < -X_WINDOW_SIZE/2 - 100 || position(1) < -Y_WINDOW_SIZE/2 - 100;
		}
		
		void draw() override {
			glColor3f(0.3f, 0.3f, 0.3f);
			glBegin(GL_POLYGON);
			// Draw the particle as a dodecagon.
			for (float t = 0.0; t < 2 * PI; t += PI/6) {
				glVertex2f(radius * cos(t) + position(0), radius * sin(t) + position(1));
			}
			glColor3f(0.3f, 0.3f, 0.3f);
			glEnd();
		}
	};
	
	AsteroidsGame() 
		: dist{0,1}
		, score{0}
		, max_score{0}
		, game_over {false}
	{
		game_over_callable = [this](){
			this->number_of_bullets = 0;
			this->number_of_particles = 0;
			this->play = Player();
			this->score = 0;
			this->max_score = 0;
			this->game_over = false;
			this->gen = std::default_random_engine();
		};
	}

	AsteroidsGame(std::function<void(void)> gameOver) 
		: dist{0,1}
		, score{0}
		, max_score{0}
		, game_over {false}
		, game_over_callable {gameOver}
	{}
	
	//Update velocities of bullets and players and positions of particles.
	void update() {
		// Update positions and velocities of all objects.
		for (int i = 0; i < number_of_particles; i++) {
			Particle& p = particles[i];

			//Check if particle is out of bounds.
			if(p.checkBounds()){
				std::copy(particles.begin()+i+1, particles.begin()+number_of_particles, particles.begin()+i);
				--i;
				--number_of_particles;
				continue;
			}

			// Collision with particle and player.
			if(distance(p, play) < p.radius + 3){
				game_over = true;
			}
			
			p.updatePosition();
		}
		
		play.updatePosition();
		if(play.checkBounds()) game_over = true;
		
		score++;
	}
	
	Eigen::MatrixXd state() {
		Eigen::MatrixXd res(16, 1);
		int number_of_angles = 8;
		
		for(int i = 0; i < number_of_angles; ++i)
		{
			double beta = -PI + (double)i*2*PI/number_of_angles;
			res(i) = -distance_to_boundary(current_game.play.position, current_game.play.orientation, beta);
			res(i+number_of_angles) = -0.5*(Eigen::Rotation2D(beta)*current_game.play.orientation).transpose() * current_game.play.velocity;
		}
		
		for (int i = 0; i < current_game.number_of_particles; ++i)
		{
			const auto& p = current_game.particles[i];
			Eigen::Vector2d d = p.position - current_game.play.position;
			double theta = angle(current_game.play.orientation, d);
			
			for(int i = 0; i < number_of_angles; ++i)
			{
				double beta = -PI + (double)i*2*PI/number_of_angles;
				double ang = theta - beta;
				double a = d.norm()*cos(ang);
				double b = p.radius*p.radius - pow(d.norm()*sin(ang),2);
				if(b > 0 && a > 0)
				{
					double distance_to_surface = a - sqrt(b);
					
					if (-distance_to_surface > res(i))
					{
						res(i) = -distance_to_surface;
						res(i+number_of_angles) = (d.transpose()/d.norm())*(p.velocity - current_game.play.velocity);
					}
				}
			}
		}
		return res;
	}
	
	void reset() {
		number_of_particles = 0;
		number_of_particles = 0;
		max_score = std::max(max_score, score);
		play = Player();
		game_over=false;
		score = 0;
	}
	
	void timer(int) {
		//display();
		if(game_over)
		{
			game_over_callable();
		}
		else update(); // Use gravity to update velocities;
	}
	
	void addRandomParticle(int size) {
		double ang = 2*PI*dist(gen);
		double beta = (PI/6)*(dist(gen)-0.5);
		double radius = size+size*dist(gen)/2;
		
		double d = std::min(abs(X_WINDOW_SIZE/2/cos(ang)), abs(Y_WINDOW_SIZE/2/sin(ang))) + 100;
		
		Eigen::Vector2d pos_{d*cos(ang), d*sin(ang)};
		Eigen::Vector2d vel_ {-3*cos(ang+beta), -3*sin(ang+beta)};
		
		if(number_of_particles < 200)
			particles[number_of_particles++] = (Particle(pos_, vel_, radius));
	}
	
	void addInitialParticle() {
		double ang = PI*dist(gen);
		Eigen::Vector2d pos_ {300*cos(ang), 300*sin(ang)};
		Eigen::Vector2d vel_ = -pos_/100;
		
		particles[0] = Particle(pos_, vel_, 100);
		particles[1] = Particle(-pos_, -vel_, 100);
		particles[2] = Particle({0, -300}, {0, 3}, 100);
		
		number_of_particles = 3;
	}
	
	//Game Objects.
	std::array<Particle, 200> particles;
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist;
	Player play;
	int score;
	int max_score;
	bool game_over;
	int number_of_bullets;
	int number_of_particles;
	std::function<void(void)> game_over_callable;
	
	static AsteroidsGame current_game;
};


}

#endif /* ASTEROIDS_GAME_HPP_ */
