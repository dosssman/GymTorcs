# Gym Torcs environment for Pip

A fork of [ugo-nama-kun's gym_torcs environment](https://github.com/ugo-nama-kun/gym_torcs) with humble improvements such as:
- Removing the need for `xautomation`: the environment can be started virtually headlessly, skipping the GUI part.
- Wrapper following the OpenAI Gym standard for environments: you can now instantiate the environment using `gym.make("Torcs-v0")`,
 which comes in handy when experimenting with stable-baselines algorithms and akin. Also adds proper action and observation spaces.
- Removes the need to manually reset the Torcs bin (due to memory leak): the interval at which the simulator is reset so as to bypass the memory leak can now be parameterized.
- Extended vtorcs-RL-color to support data recording.
- Adds race setting randomization ( opponents count, spawning location).
- Run multiple independent instance of the same environment by using the `rank` argument when creating the env.

## Potential future work
- [ ] Flesh out the installation script and include the Torcs binaries in this repository.
- [ ] Improve data recording feature.
- [ ] Better support for pixel-based training.
- [ ] Support for multi-agents and parallelization.
- [ ] More comprehensive race configuration (circuit parameterization), as well as race randomization.

## Project using GymTorcs wrapper
- [TorchRLILHybrid](https://github.com/dosssman/TorcsRLILHybrid.git) Integration with openai/baselines's GAIL, DDPG and ReMI, a versio that trades off between imitation learning of humans and a DDPG agent. A _good starting point_ on how to use this wrapper.


# Installation
## Dependencies: Torcs Racing Car Simulator Binaries

This wrapper requires a specific build of the Torcs Binaries, which can be found at https://github.com/dosssman/gym_torqs/tree/torcs_raceconfig .

The `deps_install_script.sh`script automates the installation of critical dependencies as well as the Torcs binaries themselves.
The installation script was tested on:
- Ubuntu (16.04,18.04)
- CentOS 7.2
- Arch Linux
there might be some errors occuring, since it was tested on systems where the dependencies had already been installed manually in the first place.

## Using pip and the latest commit of this repository or your own fork.
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
|rendering            | True,False       | Disables rendering in the simulation          |
|torcs_rank           | 0,1,2...         | Defines listening port. Use for parallelization|
|throttle             | True, False      | Acceleration enabled or not  |
|gear_change          | True, False      | Gear change enabled or not   |
|race_config_path     | /path/to/.../.xml| Path to the track conf file  |
|race_speed           |                  |                              |
|obs_vars             | ["angle", ...]   | Format of desired observation|
|obs_preprocess_fn    | def obs_preprocess_fn|COming soon ...           |
|obs_normalization    | True, False      | Normalize the obs. values    |
|...                  | ...              | ...                          |


## Disabling rendering for training speed up

To disable rendering during training, just need to pass `rendering=False` when instantiating the environment.
For example:
```python
env = gym.make( 'Torcs-v0', vision=False, rendering=False, obs_preprocess_fn=obs_preprocess_fn)
```
Note, however, that once disable, the agent cannot be training with pixel-based observations.
Also, despite the rendering being disabled, a window will keep popping up from time to time.
To mitigate it, use `xvfb` so the black window is render on a virtual display:
```bash
xvfb-run -a -s "-screen $DISPLAY 640x480x24" python train.py
```
with the environment variable `DISPLAY=:0`.

# Recording the players data
In case you need to record either a human's data or even that of a bot, there is a quicked hacked method that enables you to do so.
Please note that it only works for a specific set of low-level observations, such as LIDAR-like sensor data.
You might need to customize and rebuild the Torcs binary where the recording process is handled (for efficiency) to suit your need.
Data recording also need to following additional setup steps:
1. Add the data recording folder environment variable, so the Torcs binary knows where to write the data:
```
export TORCS_DATA_DIR=/home/<your username>/player_data
```
for example.

2. Make sure the Torcs binary can find the configuration files no matter where you run your training scripts from: please edit the file `vtorcs-RL-colors/src/interfaces/graphic.h` that was installed by the `deps_install_script.sh` so as to change `/home/z3r0` at Line 31 to match your path to the config files. (Default is ~/.torcs):
```
#define GR_PARAM_FILE		"/home/z3r0/.torcs/config/graph.xml" // Change this to /home/<your user name>/.torcs/config/graph.xml
```
Also change the `/home/z3r0` in the file `vtorcs-RL-colors/src/inferfaces/playerpref.h` to match your username, around line 28 to 33:
```
#define HM_DRV_FILE  		"/home/z3r0/.torcs/drivers/human/human.xml"
// dossman edit because Torcs couldn't autopmatically find the file
// #define HM_PREF_FILE		"drivers/human/preferences.xml"
#define HM_PREF_FILE		"/home/z3r0/.torcs/drivers/human/preferences.xml"
```

Make sure to rebuild the binaries by following the steps in the `deps_install_script.sh`, so as to reflect the changes.
(In case you would want to make it more seemsless, you could try to use the `getenv("HOME")` variable. The hard part is that the path to the config files is defined as a C macro, therefore making it impossible (?) to use that function to directly recover the user's home directory.)

3. Have a race configuration file that suits your recording need ready. Here is a template:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "params.dtd">


<params name="Practice">
  <section name="Header">
    <attstr name="name" val="Practice"/>
    <attstr name="description" val="Practice"/>
    <attnum name="priority" val="100"/>
    <attstr name="menu image" val="data/img/splash-practice.png"/>
    <attstr name="run image" val="data/img/splash-run-practice.png"/>
  </section>

  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="g-track-1"/>
      <attstr name="category" val="road"/>
    </section>

  </section>

  <section name="Races">
    <section name="1">
      <attstr name="name" val="Practice"/>
    </section>

  </section>

  <section name="Practice">
    <attnum name="laps" val="1"/>
    <attstr name="type" val="practice"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="yes"/>
    <attstr name="display mode" val="normal"/>
    <attstr name="display results" val="yes"/>
    <attnum name="distance" unit="km" val="0"/>
    <section name="Starting Grid">
      <attnum name="rows" val="1"/>
      <attnum name="distance to start" val="200"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" unit="km/h" val="0"/>
      <attnum name="initial height" unit="m" val="0.2"/>
    </section>

  </section>

  <section name="Drivers">
    <attnum name="maximum number" val="11"/>
    <attstr name="focused module" val="human"/>
    <attnum name="focused idx" val="1"/>
    <section name="1">
      <attnum name="idx" val="1"/>
      <attstr name="module" val="human"/>
    </section>
    <section name="2">
      <attnum name="idx" val="0"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="3">
      <attnum name="idx" val="1"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="4">
      <attnum name="idx" val="2"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="5">
      <attnum name="idx" val="3"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="6">
      <attnum name="idx" val="4"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="7">
      <attnum name="idx" val="5"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="8">
      <attnum name="idx" val="6"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="9">
      <attnum name="idx" val="7"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="10">
      <attnum name="idx" val="8"/>
      <attstr name="module" val="fixed"/>
    </section>
    <section name="11">
      <attnum name="idx" val="9"/>
      <attstr name="module" val="fixed"/>
    </section>

    <!-- Initdist for first driver-->

    <!-- Initdist for first driver-->
    <attnum name="initdist_1" val="200"/>
    <attnum name="initdist_2" val="300"/>
    <attnum name="initdist_3" val="400"/>
    <attnum name="initdist_4" val="500"/>
    <attnum name="initdist_5" val="600"/>
    <attnum name="initdist_6" val="700"/>
    <attnum name="initdist_7" val="800"/>
    <attnum name="initdist_8" val="850"/>
    <attnum name="initdist_9" val="900"/>
    <attnum name="initdist_10" val="1000"/>
    <attnum name="initdist_11" val="1150"/>

  </section>

  <section name="Configuration">
    <attnum name="current configuration" val="4"/>
    <section name="1">
      <attstr name="type" val="track select"/>
    </section>

    <section name="2">
      <attstr name="type" val="drivers select"/>
    </section>

    <section name="3">
      <attstr name="type" val="race config"/>
      <attstr name="race" val="Practice"/>
      <section name="Options">
        <section name="1">
          <attstr name="type" val="race length"/>
        </section>

        <section name="2">
          <attstr name="type" val="display mode"/>
        </section>

      </section>

   </section>
  </section>

</params>
```
To record a human player for example, make sure you have the human added as a driver in the `<section name="Drivers">` section.
Please take note of `<section name=?>` as it is used later.

4. To record the data, pass the previously created race config file to the Torcs binary directly. It will start the game, skip the menus and go to the race directly.
The complete command is as follows:
```
torcs -raceconfig $PWD/raceconfig_file.xml -rechum 0 -rectimesteplim 3600
```
- Set the argument of `-rechuman` to the *section name* the human driver was affected to.
- Set the `-rectimesteplim` argument to how many time steps you want to record. Assuming your computer can consistently simulate at 60 FPS, the data will also be recorded at the same frequency.

5. Data record format and processing.
The data is saved as three separated files:
- obs.csv: contains the observation data (distance sensor etc...). In my experiments, it was a vector of dimension 65. The data of the whole episode is saved as a single CSV line. Additional processing is require to shape it for an experience replay buffer for example. A starting point would be the following file: https://github.com/dosssman/GymTorcs/files/4681555/csv2npy.zip
- acs.csv: both steering and acceleration actions are saved (vector of dimension 2) following the same procedure as `obs.csv`.
- rews.csv: the reward is a scalar, also saved as `acs.csv` and `obs.csv`. Additional preprocessing is required, and a starting point is provided in the csv2npy.zip file mentioned in the `obs.csv` section.

6. You might want to remap the keyboard or joystick configuration to suit your preferences when recording the data.
You can do so by just running the Torcs simulator by itself, then by clicking on `Configure Players`, then on `Player`, and finally on `Controls`.
After setting your desired configuration, exit the game normally.
Next time you try to record, the control should be mapped properly, assuming the step 2 above was properly executed.
# Potential troubleshouting / Error workarounds
## AL lib: (EE) ALCplaybackOSS_open: Could not open /dev/dsp: No such device or address
Depending on your system, the audio card might not be detected by Torcs, which for some reason, absolutely requires it.
A work around was found namely here: https://ubuntuforums.org/showthread.php?t=2173702 .
Having installed the `pulseaudio-utils` package or equivalent for your system, prepend the `padsp` command everytime you run the Torcs binary, or your training scripts.
Intuitively, it will create a virtual audio device which will fool the Torcs simulator to think there is actually one.

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
