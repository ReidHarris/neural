OS := $(shell uname)

PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OBJS = asteroids.o
CXX = g++
CPPFLAGS = -Wall -O1 -std=c++2a

ifeq ($(OS), Darwin)
LDFLAGS = -framework GLUT -framework OpenGL
else
LDFLAGS = -lglut -lGLU
endif

INCFLAGS = -I $(PROJECT_ROOT) -I $(PROJECT_ROOT)../../.. -I/usr/local/include/eigen3

all:	asteroids

asteroids: $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o:	$(PROJECT_ROOT)%.cpp
	$(CXX) $(CPPFLAGS) -c $< $(INCFLAGS)

%.o:	$(PROJECT_ROOT)%.c
	$(CC) $(CFLAGS) -c $< $(INCFLAGS)

clean:
	rm -fr asteroids $(OBJS)
