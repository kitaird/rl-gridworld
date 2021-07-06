# DRL-gridworld

This project is an example for using three methods for solving the Bellman equations:
* Dynamic Programming
* Monte Carlo Method
* Temporal Difference Learning

Each approach should be implemented in the respective file and can be selected in the GUI of the GridWorld example.

##Environment dynamics

The environment acts the following way:
* Every move results in a reward of -1
* Moving into the wall will yield also a reward of -1, however the agent's position doesn't change
* Moving out of the grid will yield also a reward of -1, however the agent's position doesn't change
* When being in the goal, no more action is possible

##Configuration

The grid world layout can be adjusted in the main.py file.
It can be configured the following way:
* 'g' indicated the goal
* 'a' shows the agent in the grid world (however, this has no effect on the programm)  
* '1' indicates a wall
* '0' indicates an accessible state
* Multiple goals are possible!
* The dimension of the grid are adjustable! One can try out multiple sizes and test each algorithm with it!

#Installing and running the programm
All required packages are in resources/requirements.txt.
To install the requirements, execute 'pip install -r resource/requirements.txt'
Best pratice is to create a 'venv' with python version 3.8, then install the requirements.txt using the command above with the created venv.

Run the main.py file with python 3.8 to run the programm!

##References
The drl_board.py is based on the python package 'game2dboard' with the MIT Licence.
