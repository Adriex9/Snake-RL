Snake Game Q-learning Algorithm
This project implements a simple reinforcement learning algorithm to train an agent to play Snake game using Q-learning. 
The environment is built using a grid, the snake must move around the grid to eat fruit while avoiding collisions with the walls or itself. 
The snake grows when it eats the fruit, and the goal is to maximize the total reward by eating fruit and avoiding game over.

Algorithm Overview:
The algorithm used is a Deep Q-Learning

The game environment is a 4x4 grid (can be changed) where the snake moves and tries to eat the fruit (after the first one it spawns randomly). 
The state of the environment is represented by the grid, with different values indicating empty spaces, the snake body, the snake head, and the fruit.

The Q-Network is a neural network (using TensorFlow and Keras) is used to approximate the Q-value function. 
The network predicts the Q-values for each action (up, down, left, right) given the current state of the environment.

A replay buffer stores past experiences (state, action, reward, next state, done) and samples random batches of experiences to train the Q-network. 
This helps in breaking the correlation between consecutive samples, improving learning efficiency.

The agent starts with full exploration (epsilon = 1.0), meaning it will take random actions initially. 
Over time, epsilon decays to a minimum value (epsilon_min), and the agent begins to exploit the learned policy by selecting actions that maximize the Q-values.

The agent interacts with the environment, stores experiences in the replay buffer, and trains the Q-network using these experiences. The agent's goal is to improve its policy to maximize its total reward.

Reward System: Positive rewards are given when the snake eats the fruit, and negative rewards are assigned for invalid moves (collisions). Additional small rewards or penalties are given based on the snake's proximity to the fruit.

 I used a virtual environment to run it on visual studio code.
 A requirements.txt that list all the library needed.
