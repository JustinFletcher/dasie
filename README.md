# Distributed Aperture System for Interferometric Exploitation (DASIE)


This is a private repository for use in the the development of code related to the Distributed Aperture System for
Interferometric Exploitation (DASIE) project. This project comprises the development of a simulation environment 
compatible with the [Open AI Gym][1], a deep reinforcement leaning approach to the DASIE control problem, and relavant 
utilities.  

To install `gym` using `pip`, simply: 

``pip install gym``

This will be sufficient for most installation use cases, as evnironment registration is handled by the `gym` [registation
API][2]. To build form source instead, use:

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

Once installed, the DASIE environment included in this repostiory may be registered as follows:

```
gym.envs.registration.register(
    id='Dasie-v0',
    entry_point='dasie_gym_env.dasie:DasieEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)
```

The ID string, `Dasie-v0`, is linked to the entry point in this repository via the module `__init__` in the
`dasie_gym_evn` subdirectory.

To run a trivial demo, use:

```
import gym

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
```

Or, simply call `python gym_demo.py`.



[1]: https://gym.openai.com/docs/

[2]: https://gym.openai.com/docs/#the-registry