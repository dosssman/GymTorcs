# Gym Torcs environment for Pip

More to come ...

# Dependencies
Coming soon: Installing vtorcs-RL-color with scripts for Ubuntu, CentOS and Arch Linux (masterace)

# Installation

## Manual
Clone this repository:
```
git clone https://github.com/dosssman/GymTorcs.git && cd GymTorcs
```

Install as a python dependency:

```
pip install -e .
```

## Using Pip
```
pip install gym-torcs
```

# Usage

## Basic

Instantiate the Torcs Gym Ennvironment using gym.make("Torcs-v0"), and optionnaly passing more parameters. (Parameter list coming soon)

```
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
or whatever observatio format you might require, based on the dictionary of
observation values.

Here is an example:
```
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
```
# Return only the agent's FOV as an RGB array.

def obs_preprocess_fn( dict_obs):
    return dict_obs['img']
```
Then pass it during the environment creation:
```
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

# References
- Creating your own Gym environment (https://github.com/openai/gym/blob/master/docs/creating-environments.md)
- How to build your own pip package (https://dzone.com/articles/executable-package-pip-install)
