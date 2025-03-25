import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keyboard

# Simple snake game environment class
class SnakeEnv:
    
    
    def reset(self):
        self.size_bord=4
        self.board = np.zeros((self.size_bord, self.size_bord), dtype=int)
        self.snake = [(0, 0), (0, 1), (0, 2)]  # Reset snake position
        self.board[0, 2] = 2  # Snake's head
        self.board[0, 1] = 1  # Snake's body
        self.board[0, 0] = 1  # Snake's body
        self.fruit = (2, 2)  # Reset fruit position
        self.board[2, 2] = 10  # Fruit
        self.done = False
        return self.board.flatten()

    

    def step(self, action):
        # Define the directions: up, down, left, right
        direction_map = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}  # action: 0-up, 1-down, 2-left, 3-right
        head_x, head_y = self.snake[0]
        head=self.snake[0]
        move_x, move_y = direction_map[action]
        
        # Move snake
        new_head = (head_x + move_x, head_y + move_y)
        
        # Check if the move is valid (snake should not collide with walls or itself)
        if (new_head[0] < 0 or new_head[0] >=self.size_bord  or new_head[1] < 0 or new_head[1] >= self.size_bord or new_head in self.snake):
            self.done = True
            return self.board.flatten(), -100, self.done  # Reward is -100 for invalid move (game over)
        
        # Move snake head to the new position
        self.snake = [new_head] + self.snake #add a body to snake which is the new head
        
        # Check if snake eats fruit
        if new_head == self.fruit:
            NewFruit = True
            while NewFruit:
                self.fruit = (random.randint(0, self.size_bord-1), random.randint(0, self.size_bord-1))  # Random new fruit position
                if self.board[self.fruit]==0:
                    self.board[self.fruit] = 10
                    NewFruit=False
            reward = 100  # Reward for eating fruit
        else:
            # calculate reward for just moving
            (new_snake_head, snake_head, fruit_position)=(new_head, head, self.fruit)
            distance_before = np.linalg.norm(np.array(snake_head) - np.array(fruit_position))
            distance_after = np.linalg.norm(np.array(new_snake_head) - np.array(fruit_position))
            if distance_after < distance_before:
                reward = +1  # Earn point for being closer
            elif distance_after > distance_before:
                reward = -1  # Penalized for being going less close
            else:
                reward = 0
              
            self.snake=self.snake[:-1]      #remove extra body part as the snake didn't eat a fruit
        if len(self.snake)==self.size_bord*self.size_bord:
            print("you won")
        # Update the board (0 for empty, 1 for snake, 2 for snake's head, 10 for fruit)
        self.board = np.zeros((self.size_bord, self.size_bord), dtype=int)
        head=True
        for x, y in self.snake:
            if head:
                self.board[x, y] = 2
                head=False
            else :
                self.board[x, y] = 1
        self.board[self.fruit] = 10
        
        return self.board.flatten(), reward, self.done  # Return flattened board as state

    

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add(self, experience):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)  # Remove the oldest experience
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Create custom snake environment
env = SnakeEnv()

# Q-network for deep Q-learning
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# Hyperparameters
num_actions = 4  # Up, Down, Left, Right
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Start with full exploration
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 10000
batch_size = 64
buffer_size = 1000

# Initialize Q-network, optimizer, and replay buffer
q_network = QNetwork(num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate)
buffer = ReplayBuffer(buffer_size)

# Training function
def train_step(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    with tf.GradientTape() as tape:
        q_values = q_network(states)
        q_values_next = q_network(next_states)
        
        max_q_values_next = np.max(q_values_next, axis=1)
        targets = rewards + gamma * (1 - dones) * max_q_values_next
        
        one_hot_actions = tf.one_hot(actions, num_actions)
        predicted_q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)

        loss = tf.reduce_mean((targets - predicted_q_values) ** 2)
    
    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))



# Visualization 
def see_snake():
    state = env.reset()
    done = False
    fig, ax = plt.subplots(figsize=(8, 8))
    
    while not done:
        # Choose action based on the trained Q-network
        q_values = q_network(np.array([state], dtype=np.float32))
        action = np.argmax(q_values)

        next_state, reward, done = env.step(action)
        state = next_state
        
        # Clear the axes and redraw the board
        ax.clear()
        
        # Set up the grid and labels
        ax.set_xticks(np.arange(0, env.size_bord, 1))
        ax.set_yticks(np.arange(0, env.size_bord, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, env.size_bord)
        ax.set_ylim(0, env.size_bord)    
        board_now = env.board
        for i in range(board_now.shape[0]):
            for j in range(board_now.shape[1]):
                value = board_now[i, j]

                # Define colors based on the value
                if value == 0:
                    color = 'white'  # Empty space
                    edge_color = 'black'  # Black corners for empty space
                elif value == 1:
                    color = 'green'  # Snake body
                    edge_color = 'black'  # Black border for snake
                elif value == 2:
                    color = 'limegreen'  # Snake head
                    edge_color = 'black'  # Black border for snake
        
                elif value == 10:
                    color = 'red'  # Fruit
                    edge_color = 'black'  # Black border for fruit
            
                # Add squares with custom color and black edge
                ax.add_patch(patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor=edge_color, facecolor=color))
        ax.set_aspect('equal')
        plt.draw()  # Redraw the plot
        plt.pause(0.5)  # Pause to visualize the board

    plt.close(fig)  # Close the plot after the game is done


    # Main training loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step =0
    
    while not done:
        if step==300:
            done=True
        # Exploration vs exploitation
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            q_values = q_network(np.array([state], dtype=np.float32))
            action = np.argmax(q_values)
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Take the action and get the next state, reward, and done flag
        next_state, reward, done = env.step(action)
        step+=1
        
        # Add the experience to the replay buffer
        buffer.add((state, action, reward, next_state, done))
        
        # Train the model using experience replay
        if len(buffer.buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            train_step(batch)

        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
    if episode %100==0 :
        see_snake()
    

def play_snake():
    state = env.reset()
    done = False
    fig, ax = plt.subplots(figsize=(8, 9))
    action = 2
    while not done:
        # Choose action (azerty keybord)
        if keyboard.is_pressed('q'):  # if 'q' is pressed  
            action = 2  # left
        elif keyboard.is_pressed('z'):  # if 'z' is pressed
            action = 0  # up
        elif keyboard.is_pressed('s'):  # if 's' is pressed
            action = 1  # down
        elif keyboard.is_pressed('d'):  # if 'd' is pressed
            action = 3  # right
        else: 
            continue
        next_state, reward, done = env.step(action)
        state = next_state
        
        # Clear the axes and redraw the board
        ax.clear()
        
        # Set up the grid and labels
        ax.set_xticks(np.arange(0, env.size_bord, 1))
        ax.set_yticks(np.arange(0, env.size_bord, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, env.size_bord)
        ax.set_ylim(0, env.size_bord)
        board_now = env.board
        for i in range(board_now.shape[0]):
            for j in range(board_now.shape[1]):
                value = board_now[i, j]

                # Define colors based on the value
                if value == 0:
                    color = 'white'  # Empty space
                    edge_color = 'black'  # Black corners for empty space
                elif value == 1:
                    color = 'green'  # Snake body
                    edge_color = 'black'  # Black border for snake
                elif value == 2:
                    color = 'limegreen'  # Snake head
                    edge_color = 'black'  # Black border for snake
        
                elif value == 10:
                    color = 'red'  # Fruit
                    edge_color = 'black'  # Black border for fruit
            
                # Add squares with custom color and black edge
                ax.add_patch(patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor=edge_color, facecolor=color))
        ax.set_aspect('equal')
        plt.draw()  # Redraw the plot  
        plt.pause(0.02)
    plt.close(fig)  # Close the plot after the game is done
