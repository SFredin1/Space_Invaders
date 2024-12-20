import keras
import gymnasium as gym
import ale_py
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

# Register Atari environments
gym.register_envs(ale_py)

# Path to trained model
model_file = 'Lab_Space-Invaders/Space-Invaders_q_model170.keras'
agent = keras.models.load_model(model_file)

# Set up Atari environment
env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='human')
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)

# Reset environment and get initial state
state, _ = env.reset()
done = False

while not done:
    # Convert state to a tensor
    state_tensor = keras.ops.convert_to_tensor(state, dtype='float32')
    # Ensure the shape is (84, 84, 4)
    state_tensor = keras.ops.transpose(state_tensor, [1, 2, 0])
    # Add batch dimension
    state_tensor = keras.ops.expand_dims(state_tensor, axis=0)
    # Get action probabilities from the trained agent
    action_probs = agent(state_tensor, training=False)
    # Select the action with the highest Q-value
    action = keras.ops.argmax(action_probs[0]).numpy()
    # Step the environment with the selected action
    state, reward, done, _, _ = env.step(action)

env.close()