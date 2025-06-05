# 🐍 Snake Game with Deep Q-Learning

This project implements a simple Snake game AI agent using Deep Q-Learning. The environment is custom-built on a grid, and the agent learns to navigate it, eat fruits, and avoid collisions with walls or itself using reinforcement learning principles.

![SnakeRL](https://github.com/user-attachments/assets/70fb16b9-8986-4500-ae5c-eb9427bc8ccc)

---

## Project Overview

- **Environment**: 6x6 grid (modifiable via `bord_size`)
- **Agent Goal**: Eat fruit, grow in size, avoid death
- **Library Stack**: `TensorFlow`, `NumPy`, `Matplotlib`, `keyboard`
- **Training**: Deep Q-Network with experience replay and epsilon-greedy policy

---

## Algorithm Highlights

- **State Space**: Flattened grid where:
  - `0` = empty cell
  - `1` = snake body
  - `2` = snake head
  - `10` = fruit

- **Actions**:
  - `0` = up
  - `1` = down
  - `2` = left
  - `3` = right

- **Q-Network**:
  - 2 hidden layers (64 neurons, ReLU)
  - Fully connected output layer (4 actions)
  - Trained using Mean Squared Error loss

- **Replay Buffer**:
  - Stores `(state, action, reward, next_state, done)`
  - Helps stabilize training by breaking correlation between samples

---

## 🏆 Reward System

| Action                         | Reward  |
|-------------------------------|---------|
| Eating a fruit                | **+100** |
| Collision (game over)         | **-100** |
| Moving closer to the fruit    | **+1**   |
| Moving away from the fruit    | **-2**   |

This helps the agent learn to:
- Navigate intelligently toward fruit
- Avoid risky or non-optimal moves

---

## 🧪 Training

- **Episodes**: 10,000
- **Batch Size**: 64
- **Discount Factor (γ)**: 0.99
- **Epsilon Decay**: Starts at 1.0 → decays to 0.01
- **Model**: Trained in TensorFlow using `Adam` optimizer

Every 250 episodes, the agent's performance is visualized using a custom plotting function.

---

## 🎮 Playing the Snake (Manual Mode)

The game can also be played manually using the keyboard 
 (Was very usefull to test rules of the game)

| Key (AZERTY) | Action  |
|--------------|---------|
| `Z`          | Up      |
| `S`          | Down    |
| `Q`          | Left    |
| `D`          | Right   |

Run `play_snake()` to launch manual mode.

---

## Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
