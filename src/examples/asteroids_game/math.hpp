//
//  math.hpp
//  Asteroids
//
//  Created by Reid Harris on 12/9/20.
//  Copyright © 2020 Reid Harris. All rights reserved.
//

#ifndef ASTEROIDS_MATH_HPP_
#define ASTEROIDS_MATH_HPP_

#include <Eigen/Dense>
#include <Eigen/Geometry>


#define X_WINDOW_SIZE 800
#define Y_WINDOW_SIZE 700
#define PI 3.14159265

namespace asteroids
{

Eigen::Vector2d counterclockwise_normal(const Eigen::Vector2d& orient) {
	//(1,0) becomes (0,1) becomes (-1,0), etc.
	Eigen::Vector2d res {-orient(1), orient(0)};
	return res;
}

double cross_product(Eigen::Vector2d& v1, Eigen::Vector2d& v2) {
	// <0 if v1 lies counterclockwise to v2
	return v1(0)*v2(1) - v1(1)*v2(0);
}

bool counterclockwise_rotation_to_from(Eigen::Vector2d& v1, Eigen::Vector2d& v2) {
	// Returns true if the v2 is obtained from v1 by a counterclockwise rotation of v1 by an angle <= pi.
	// Equivalently, if v2 "lies to the left" of v1.
	return v1(0)*v2(1) - v1(1)*v2(0) >= 0;
}

double angle(Eigen::Vector2d& axis, Eigen::Vector2d& v) {
	//axis is assumed to be unit vector
	//returns an angle in (-pi, pi]
	if(v.norm() < 0.005) return 0;
	Eigen::Vector2d v_unit = v/v.norm();
	return acos(v_unit.transpose()*axis) * ((counterclockwise_rotation_to_from(axis, v_unit)) ? 1 : -1 );
}

double distance_to_boundary(Eigen::Vector2d& pos, Eigen::Vector2d& dir, double ang) {
	Eigen::Vector2d new_direction = Eigen::Rotation2D<double>(ang)*dir;
	double x_max = 1000, y_max = 1000;
	if(abs(new_direction(0)) > 0.005) x_max = std::max( (X_WINDOW_SIZE/2 - pos(0))/new_direction(0), (-X_WINDOW_SIZE/2 - pos(0))/new_direction(0) );
	if(abs(new_direction(1)) > 0.005) y_max = std::max( (Y_WINDOW_SIZE/2 - pos(1))/new_direction(1), (-Y_WINDOW_SIZE/2 - pos(1))/new_direction(1) );
	return std::min(x_max, y_max);
}
}
#endif /* ASTEROIDS_MATH_HPP_ */
