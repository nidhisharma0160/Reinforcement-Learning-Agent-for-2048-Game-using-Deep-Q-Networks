# Reinforcement-Learning-Agent-for-2048-Game-using-Deep-Q-Networks
I have built a Deep Q-Network (DQN) agent that learns to play 2048 using a CNN Q-network, replay memory, target network, and epsilon-greedy exploration, with a Tkinter GUI autoplay demo.


This project trains an AI agent to play the puzzle game **2048** using **Deep Q-Learning (DQN)**.  
Instead of hand-written heuristics, the agent learns by interacting with the game environment, collecting experience, and updating a CNN-based Q-network to choose the best action (Up, Left, Right, Down).  

## Key features
- Custom **2048 game environment** (move logic, merge rules, scoring, terminal state)
- **State encoding**: board -> one-hot tensor of shape **(1, 4, 4, 16)** (tile values as powers of 2)
- **CNN-based Q-network** with parallel conv paths to capture horizontal and vertical patterns
- **Epsilon-greedy exploration** for exploration vs exploitation
- **Replay memory** + random minibatch training for stable learning
- **Target network** updated periodically to stabilize Q-targets
- **Reward shaping** using:
  - bonus when max tile increases (log-based)
  - reward for increasing empty cells
- **Checkpointing** (model + scores/loss) for long Colab runs
- **Tkinter GUI autoplay** using the trained model

## System overview
High-level pipeline:
1. Game generates state `s`
2. Encode `s` to one-hot tensor
3. Q-network predicts Q(s, a) for 4 actions
4. Select action via epsilon-greedy (or greedy for evaluation)
5. Execute action, observe reward and next state
6. Store (s, a, r, s') in replay memory
7. Sample minibatches and train with Bellman targets using a target network

## Tech stack
- Python
- TensorFlow / Keras
- NumPy
- Tkinter (GUI autoplay)
- Matplotlib (training curves)

