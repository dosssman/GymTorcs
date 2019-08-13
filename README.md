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

Instantiate the Torcs Gym Ennvironment using gym.make("Torcs-v0"), and optionnaly passing more parameters. (Parameter list coming soon)

```
import gym

# Gym Torcs dep.
import gym_torcs

# A prototype of Agent randomly sampling actions
from gym_torcs.sample_agent import Agent

try:
  # Instantiate the environment
  env = gym.make( "Torcs-v0")

  o ,r, done = env.reset(), 0., False
  while not done:
    action = agent.act( o, r, done, False)
    o, r, done, _ = env.step( action)

excpet Execption as e:
  print( e)

finally:
  env.end()

```

# References:
- Creating your own Gym environment (https://github.com/openai/gym/blob/master/docs/creating-environments.md)
- How to build your own pip package (https://dzone.com/articles/executable-package-pip-install)
