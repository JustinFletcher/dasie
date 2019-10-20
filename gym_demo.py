import gym

# TODO: Extract dasie to it's own module.
gym.envs.registration.register(
    id='Dasie-v0',
    entry_point='dasie_gym_env.dasie:DasieEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)


env = gym.make('Dasie-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()