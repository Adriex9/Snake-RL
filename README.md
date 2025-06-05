# 🐍 Snake Game – Q-Learning Reinforcement Learning Agent

This project implements a **Q-Learning algorithm** to train an agent to play the classic **Snake Game** using deep reinforcement learning techniques.

The environment is a grid-based board where the snake must eat fruits to grow while avoiding collisions with the walls and itself. The objective is to **maximize cumulative reward** through learned behavior.

---

## 🎮 Game Environment

- Grid-based board (default: **4x4**, easily configurable)
- Snake moves toward randomly spawning fruit
- The snake grows when eating fruit
- Episode ends upon collision (wall or self)

![SnakeRL](https://github.com/user-attachments/assets/70fb16b9-8986-4500-ae5c-eb9427bc8ccc)

---

## 🤖 Algorithm Overview

The project uses **Deep Q-Learning** to teach the agent how to play the game.

### 🧠 Q-Network

- Built with **TensorFlow** and **Keras**
- Inputs: Current state (grid representation)
- Outputs: Q-values for actions `[up, down, left, right]`

### 🌀 State Representation

- Grid with encoded values for:
  - Empty space
  - Snake body
  - Snake head
  - Fruit

### 🔁 Replay Buffer

- Stores experiences: `(state, action, reward, next_state, done)`
- Random sampling breaks temporal correlations
- Enables **efficient, stable learning**

---

## 🎯 Training Strategy

- **Exploration–Exploitation Tradeoff**:
  - Starts with `epsilon = 1.0` (full exploration)
  - Gradual decay toward `epsilon_min`
  - Eventually favors best learned action (exploitation)

### 🏆 Reward System

The Q-learning agent is guided by a custom reward structure to encourage optimal gameplay behavior:

| Action                         | Reward  |
|-------------------------------|---------|
| Eating a fruit                | **+100** |
| Collision (game over)         | **-100** |
| Moving closer to the fruit    | **+1**   |
| Moving away from the fruit    | **-2**   |

---

## 🛠️ Development Environment

- Developed and tested using **Visual Studio Code**
- Uses a **Python virtual environment**
- All required libraries are listed in `requirements.txt`

### 📦 Dependencies

