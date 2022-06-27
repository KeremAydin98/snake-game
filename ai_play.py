import matplotlib.pyplot as plt
import time
import gym

env = gym.make("gym_snake:snake-v0")

env.reset()
env.render("human")

for i in range(100):
    env.render("human")
    action = env.action_space.sample()
    img, reward, done, info = env.step(action)
    time.sleep(0.1)


plt.figure()
plt.imshow()