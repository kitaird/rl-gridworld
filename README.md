# RL-Gridworld: A RL-Learning Environment

Welcome to the __RL-Gridworld__, an open-source resource designed for learning and experimenting with various paradigms in reinforcement learning (RL). 
This library provides a versatile gridworld environment that can be easily extended and customized, making it an ideal tool for both beginners and experienced practitioners.

This project is an example for using three methods for solving the Bellman equations (Prediction & Control):
* Dynamic Programming
* Monte Carlo Method
* Temporal Difference Learning

All prediction and control approaches can be implemented in the respective file and can be selected in the GUI of the GridWorld example.

![Agent Starting State](example-images/Agent_Starting_State.png)

## Features
* __Gymnasium Interface__: The gridworld environment is designed to be compatible with the popular RL-interface [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), allowing users to transition their gained knowledge easily to more sophisticated DRL-libraries.

* __Extensible Gridworld Environment__: At the core of this library is the gridworld environment, a simple yet powerful tool for demonstrating key concepts in RL. 
Users can easily modify and extend this environment to suit their learning and research needs.

* __Support for Multiple RL Paradigms__: The library is built to demonstrate a variety of reinforcement learning techniques, including:
  * Dynamic Programming
  * On-Policy Monte-Carlo Prediction
  * On-Policy Monte-Carlo Control (w and w/o exploring starts)
  * TD Prediction TD(0)
  * On-Policy TD Control Sarsa

## Environment dynamics

The environment acts the following way:
* Every move results in a reward of `-1`
* Moving into the wall will also yield a reward of `-1`, however the agent's position doesn't change
* Moving out of the grid will also yield a reward of `-1`, however the agent's position doesn't change
* When being in the goal, no more action is possible (terminal state)

## Configuration

The grid world layout can be adjusted in the `__main__.py` file.
It can be configured the following way:
* _g_ indicates the goal
* _a_ indicates the agent's starting position
* _1_ indicates a wall
* _0_ indicates an accessible state
* Multiple goals are possible!
* The dimensions of the grid are adjustable! One can try out multiple sizes and test each algorithm with it!

## Solutions for algorithms
The solution for the algorithms is in the directory `agents/solutions`. These are currently used in the main.py in order to see that the example is working.
When assigning the task to the students, the solutions directory should be removed and the regular algorithms (`dp_control.py`, `mc_prediction.py`, `mc_control.py`, `td_prediction.py`, `td_control.py`) referenced in the `__main__.py`.

# Installing and running the program
All required packages are in resources/requirements.txt.
To install the requirements, execute `pip install -r resources/requirements.txt`.
Best practice is to create a 'venv' with python version 3.9, then install the `resources/requirements.txt` using the command above with the created venv.

Run the `__main__.py` file with python 3.9 to run the program!

## Example images
Here are some examples of the project with implemented algorithms:

### Empty Gridworld
![Gridworld](example-images/Gridworld.png)

### Empty Gridworld with Agent starting position
![Agent Starting State](example-images/Agent_Starting_State.png)

### Initialized Policy
![Initialized Policy](example-images/Initialized_Policy.png)

### Initialised Action Values
![Initialized Action Values](example-images/Initialized_Action_Values.png)

### Converged Action Values
![Converged Action Values](example-images/Converged_Action_Values.png)

### Optimal Policy using Dynamic Programming
![Optimal Policy: DP](example-images/Optimal_Policy_DP.png)

## References
The rl_board.py is based on the source code ot the python package `game2dboard` which uses the provided [game2dboard-MIT-Licence](https://github.com/kitaird/rl-gridworld/blob/develop/resources/game2dboard-LICENSE.txt) under `resources/game2dboard-LICENSE.txt`.
The initial `board.py` from [mjbrusso/game2dboard](https://github.com/mjbrusso/game2dboard) was extended to contain additional buttons and logic for the purpose of this project.

## License
This project is licensed under the MIT License - see the [MIT-Licence](https://github.com/kitaird/rl-gridworld/blob/develop/LICENSE.txt) file for details.

## Citation
If you find this project helpful and use it, please cite it like so:
```bibtex
@misc{gashi2023rl-gridworld,
      title={RL-Gridworld: A RL-Learning Environment},
      author={Adriatik Gashi},
      institution = {Darmstadt University of Applied Sciences},
      howpublished = {\textsc{url:}~\url{https://github.com/kitaird/rl-gridworld}},
      year={2023}
}
```