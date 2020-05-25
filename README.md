# Gym Torcs environment for Pip

A fork of [ugo-nama-kun's gym_torcs environment](https://github.com/ugo-nama-kun/gym_torcs) with humble improvements such as:
- Removing the need for `xautomation`: the environment can be started virtually headlessly, skipping the GUI part.
- Wrapper following the OpenAI Gym standard for environments: you can now instantiate the environment using `gym.make("Torcs-v0")`,
 which comes in handy when experimenting with stable-baselines algorithms and akin. Also adds proper action and observation spaces.
- Removes the need to manually reset the Torcs bin (due to memory leak): just define an interval and the library takes care of the rest.
- Support observation customisation.
- Extended vtorcs-RL-color to support data recording.
- Adds race setting randomization ( opponents count, spawning location).
- Run multiple independent instance of the same environment by using the `rank` argument when creating the env.

## Potential future work
[] Flesh out the installation script and include he Torcs binaries in this repository.
[] More general support for data recording.
[] Better support for pixel-based training.
[] Support for multi-agents and parallelization.
[] More comprehensive circuit parameterization / randomization.

# Installation
## Dependencies: Torcs Racing Car Simulator Binaries

This wrapper requires a specific build of the Torcs Binaries, which can be found at https://github.com/dosssman/gym_torqs/tree/torcs_raceconfig .

The `deps_install_script.sh`script automates the installation of critical dependencies as well as the Torcs binaries themselves.
The installation script was tested on:
- Ubuntu (16.04,18.04)
- CentOS 7.2
- Arch Linux
there might be some errors occuring, since it was tested on systems where the dependencies had already been installed manually in the first place.

## Using pip and the latest commit of this repository
```bash
pip install -e git+https://github.com/dosssman/GymTorcs#egg=gym_torcs
```
If you use Conda for virtual environment management, not that you can also add the line `- git+https://github.com/dosssman/GymTorcs#egg=gym_torcs` to have it install it in a similar fashion.

## Manual (Recommended for in-depth customization)
Clone this repository, or your own fork of it:
```bash
git clone https://github.com/dosssman/GymTorcs.git && cd GymTorcs
```
then install the local version

```bash
pip install -e .
```

## Using PyPi package (usually lagging behind, hence not recommended)
```bash
pip install gym-torcs
```
Note: This git repository is likely to be more up to date.

# Usage

## Basic

Instantiate the Torcs Gym Ennvironment using gym.make("Torcs-v0"), and optionnaly passing more parameters. (Parameter list coming soon)

```python
import gym

# Gym Torcs dep.
import gym_torcs

try:
  # Instantiate the environment
  env = gym.make( "Torcs-v0")

  o ,r, done = env.reset(), 0., False
  while not done:
    action = np.tanh(np.random.randn(env.action_space.shape[0]))
    o, r, done, _ = env.step( action)

excpet Execption as e:
  print( e)

finally:
  env.end()

```
## Realistic

By default, the observations are provided as a dictionay of values.
Therefore, to fit the observation format to your experiement requirements,
use the `obs_preprocess_fn` parameter of gym.make() to pass a customized
preprocessing function.

The function should take one argument called `dict_obs` and return an array,
or whatever observation format you might require, based on the dictionary of
observation values.

Here is an example:
```python
# Builds an array with observations such as angle, track, speeds, etc...
def obs_preprocess_fn(dict_obs):
     return np.hstack((dict_obs['angle'],
         dict_obs['track'],
         dict_obs['trackPos'],
         dict_obs['speedX'],
         dict_obs['speedY'],
         dict_obs['speedZ'],
         dict_obs['wheelSpinVel'],
         dict_obs['rpm'],
         dict_obs['opponents']))
```
Another one:
```python
# Return only the agent's FOV as an RGB Image

def obs_preprocess_fn( dict_obs):
    return dict_obs['img']
```
Then pass it during the environment creation:
```python
env = gym.make( 'Torcs-v0', vision=vision, obs_preprocess_fn=obs_preprocess_fn)
```

## Parameter list:

|      Parameter      |      Values      |            Desc.             |
|---------------------|------------------|------------------------------|
|throttle             | True, False      | Acceleration enabled or not  |
|gear_change          | True, False      | Gear change enabled or not   |
|race_config_path     | /path/to/.../.xml| Path to the track conf file  |
|race_speed           |                  |                              |
|obs_vars             | ["angle", ...]   | Format of desired observation|
|obs_preprocess_fn    | def obs_preprocess_fn|COming soon ...           |
|obs_normalization    | True, False      | Normalize the obs. values    |
|...                  | ...              | ...                          |

# References and Aknowledgement
- Creating your own Gym environment (https://github.com/openai/gym/blob/master/docs/creating-environments.md)
- How to build your own pip package (https://dzone.com/articles/executable-package-pip-install)
- Original Gym Torcs environment (https://github.com/ugo-nama-kun/gym_torcs) (Deep gratitude).

# Citation
In case you would like to cite this repository, please do so using the following BIB data.
```
@misc{GymTorcs,
  title={GymTorcs: An OpenAI Gym-style wrapper for the Torcs Racing Car Simulator},
  author={Rousslan Fernand Julien Dossa},
  year={2018}
}
```
