# Distributed Aperture System for Interferometric Exploitation (DASIE)

This repository supports the development of code related to the
Distributed Aperture System for Interferometric Exploitation (DASIE) project.
This project comprises the development of a simulation environment compatible 
with the [Open AI Gym][1], a deep reinforcement learning approach to the DASIE 
visuomotor control problem, and a static planning approach in which DASIE
control is  formulated as an electromechanical metasurface design problem which
is in turn solved by gradient descent.

## Visuomotor Control

To install `gym` and other dependencies using `pip`, simply: 

``pip install gym==0.21.0 Pillow matplotlib joblib hcipy pyglet``

This will be sufficient for most installation use cases, as environment registration is handled by the `gym` [registation
API][2]. To build `gym` from source instead, use:

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

Or, simply call `python dasie\visuomotor\run_dasie_via_gym.py`.

## Open-Loop Planning

To install the MLFlow experiment management and Pyrallis configuration libaries:
```python
pip install mlflow pyrallis 
```

To start an MLFlow tracking server for viewing experiments, runs, and artifacts, navigate to your local `mlruns` directly and run:
```bash
mlflow ui
```

In TF 2.3+, invoke the following command to run the existing training routine
with sequential Zernike modeling per subaperture.

```
python ./dasie/planning/differentiable_dasie_graph.py  --run_name=debug --num_steps=4096 --spatial_quantization=512 --batch_size=16 --recovery
_model_filter_scale=2 --num_exposures=2 --save_plot --dataset_name=speedplu
s --dataset_root=../data --gpu_list=0 --plot_periodicity=32 --aperture_diam
eter_meters=20.0 --num_zernike_indices=15 --num_subapertures=15 --crop --ha
damard_image_formation --zernike_debug
```

To run the standard training routine, issue the same command without the
 `--zernike_debug` flag. 

By default, `train.py` executes a model saving step after each validation. 
Saved models are stored as JSON files on disk, and include the `kwargs` used to
instantiate and train the model. An example saved model can be found at 
`./dasie/resources/model_save_0.json`.

For a DASIE model, inference corresponds to image recovery from a collected 
ensemble of images from a distributed aperture system. To restore a model and
perform image recovery, just invoke `python ./dasie/planning/recover.py`. The 
default flags for this script will restore the example saved model, and will 
attempt to recover an image from the ensemble stored at 
`./dasie/resources/example_ensemble`. The saved model is untrained, and as such
will not produce anything interesting, but this example exercises the full save,
restore, and recover (i.e., inference) workflow. Flags are present within 
`recover.py` that enable adaptation and reuse beyond the example.

Finally, we must be able to export a learned articulation plan. This is 
straightforward, because the DASIE saved model format uses the TensorFlow
variable name of each variable as the key for that variable's values. When 
combined with the `DASIEModel` class name scoping, this allows for simple 
parsing of the Zernike coefficient values for each exposure and subaperture. 
One such parser is implemented in `get_plan.py`, which parses the saved 
dictionary and constructs a numpy array. To print this array, invoke
`python ./dasie/planning/get_plan.py`. Again, this script operates on the
saved model in `./dasie/resources/model_save_0.json` by default, but exposes 
flags for easy reuse.




[1]: https://gym.openai.com/docs/

[2]: https://gym.openai.com/docs/#the-registry