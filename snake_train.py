import numpy as np
import random
from collections import deque
import gym
import time
import tensorflow as tf


env = gym.make("gym_snake:snake-v0")

# Observations
num_observations = env.observation_space.shape[0]

# Actions
num_actions = env.action_space.n

# input_shape=4 ANN ---> neurons==actions
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(1,4), activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),

    # Neurons == action_space
    tf.keras.layers.Dense(num_actions)

])

target_model = tf.keras.models.clone_model(model)

# Hyperparameters
EPOCHS = 1000
epsilon = 1.0
EPSILON_REDUCE = 0.995
GAMMA = 0.001
BATCH_SIZE = 32


def epsilon_greedy_action_selection(model, epsilon, observation):

    if np.random.random() > epsilon:

        prediction = model.predict(observation)
        action = np.argmax(prediction)

    else:

        action = np.random.randint(0, env.action_space.n)

    return action

# Removes its oldest element when the length of maxlen is surpassed
# We will use this as our buffer
replay_buffer = deque(maxlen=20000)
update_target_model = 10


def replay(replay_buffer, batch_size, model, target_model):

    if len(replay_buffer) < batch_size:

        return

    samples = random.sample(replay_buffer, batch_size)

    target_batch = []

    zipped_samples = list(zip(*samples))

    states, actions, rewards, new_states, dones = zipped_samples

    targets = target_model.predict(np.array(states))

    q_values = model.predict(np.array(new_states))

    for i in range(batch_size):

        q_value = max(q_values[i][0])

        target = targets[i].copy()

        if dones[i]:
            target[0][actions[i]] = rewards[i]

        else:
            target[0][actions[i]] = rewards[i] + q_value*GAMMA

        target_batch.append(target)

    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)


def update_model_handler(epoch, update_target_model, model, target_model):

    if epoch > 0 and epoch % update_target_model == 0:

        target_model.set_weights(model.get_weights())


model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam())

best_so_far = 0

for epoch in range(EPOCHS):

    observation = env.reset()

    observation = observation.reshape([1,4])

    done = False

    points = 0

    while not done:

        action = epsilon_greedy_action_selection(model, epsilon, observation)

        next_observation, reward, done, info = env.step(action)

        next_observation = next_observation.reshape([1,4])

        replay_buffer.append((observation, action, reward, next_observation, done))

        observation = next_observation
        points += 1

        replay(replay_buffer, BATCH_SIZE, model, target_model)

        print("points", points)

    epsilon *= EPSILON_REDUCE

    update_model_handler(epoch, update_target_model, model, target_model)

    if points > best_so_far:

        best_so_far = points

        model.save("best one.h5")

    print(f"Epoch: {epoch}, POINTS: {points}, eps: {epsilon}, BSF: {best_so_far}")
