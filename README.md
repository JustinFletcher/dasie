# Distributed Aperture System for Interferometric Exploitation (DASIE)

In TF 2.3+, invoke the following command to run the existing training routine with sequential Zernike modeling per subaperture.

```
python ./src/planning/differentiable_dasie_graph.py  --run_name=debug --num_steps=4096 --spatial_quantization=512 --batch_size=16 --recovery
_model_filter_scale=2 --num_exposures=2 --save_plot --dataset_name=speedplu
s --dataset_root=../data --gpu_list=0 --plot_periodicity=32 --aperture_diam
eter_meters=20.0 --num_zernike_indices=15 --num_subapertures=15 --crop --ha
damard_image_formation --zernike_debug
```

To run the standard training routine, issue the same command without the `--zernike_debug` flag.

Eventually, we'll reconstruct the workflow below for a SDMUU formulation

This is a private repository for use in the development of code related to the Distributed Aperture System for
Interferometric Exploitation (DASIE) project. This project comprises the development of a simulation environment 
compatible with the [Open AI Gym][1], a deep reinforcement learning approach to the DASIE control problem, and relevant 
utilities.  

To install `gym` using `pip`, simply: 

``pip install gym``

This will be sufficient for most installation use cases, as environment registration is handled by the `gym` [registation
API][2]. To build form source instead, use:

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

Once installed, the DASIE environment included in this repo may be registered as follows:

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

Or, simply call `python src\visuomotor\run_dasie_via_gym.py`.



[1]: https://gym.openai.com/docs/

[2]: https://gym.openai.com/docs/#the-registry