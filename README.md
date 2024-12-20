# Space_Invaders
## Project: ATARI-2600 Emulator, Space Invaders AI with Deep Q-Learning

## Overview
This project demonstrates the use of Deep Q-Learning to train an AI agent to play the Atari 2600 game "Space Invaders". The implementation uses TensorFlow for building the model, and the Gymnasium library for enviroment simulation.

## Features
* **Deep Q-Learning Algorithm**: Implements a neural network-based Q-learning agent.
* **Enviroment Interaction**: Simulates gameplay using Gymnasiums Atari enviroment.
* **Pre-trained Model**: Includes a trained model for evaluation and demonstration purposes.
* **Visulization**: Displays the gameplay using the enviroment rendering.

## Files
1. space_invaders.py

This python script loads a pre-trained Deep Q-Learning model and uses it to play the game. The key functionalitys are:
* Loading the trained model (in this case "Space-Invaders_q_model170.keras").
* Initializing the "SpaceInvadersNoFrameskip-v4" enviroment with preprocessing (grayscale conversion and frame stacking).
* Visualizing the agents performance with rendered gameplay.

2. space_invaders.ipynb

This Jupyter Notebook contains step-by-step implementation details for training the Deep Q-Learning agent. The key sections are:
* Enviroment setup: Defining a convolutional neural network for Q-Learning
* Training loop: Training the agent using replay memory and epsilon-greedy exploration.
* Saving the Model: Storing the trained model. 

## How to Use
**Prerequisites**
* Python3.x
* TensorFlow
* Gymnasium and ALE-py libraries

**Running the Project**
1. **Evaluate the Pre-Trained Model**:
    * Run the space_invaders.py script
    * The script will load the pre-trained model and simulate/render gameplay.
2. **Training the model**:
    * In space_invaders.ipynb jupyter notebook, follow/run the steps/code cells in order to train the model. 
    * Save the (best) trained model.

## Model Details
* **Input Shape**: (84,84,4) - 4 consecutive grayscale frames stacked.
* **Architecture**:
    * Convolutional layers for feature extraction.
    * Fully connected layers for decision making.
    * Output layer with linear activation for Q-values.
