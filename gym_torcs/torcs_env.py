import gym
from gym import spaces
import numpy as np
# from os import path
# import baselines.ddpg_torqs.snakeoil3_gym as snakeoil3
import gym_torcs.snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
### TODO: Get out of the way: os
import os
import subprocess
import psutil
import time
import math
import random
from xml.etree import ElementTree as ET

DEF_BOX_DTYPE = np.float32

class TorcsEnv( gym.Env):
    terminal_judge_start = 50.  # Speed limit is applied after this step
    termination_limit_progress = 1  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 300.

    initial_reset = True

    # Customized to accept more params
    def __init__(self, vision=False,
        throttle=False,
        gear_change=False,
        race_config_path=None,
        race_speed=1.0,
        rendering=True,
        damage=False,
        lap_limiter=1,
        recdata=False,
        noisy=False,
        rec_episode_limit=1,
        rec_timestep_limit=3600,
        rec_index=0,
        hard_reset_interval=11,
        randomisation=False,
        profile_reuse_ep=500,
        rank=0,
        obs_vars=None,
        obs_preprocess_fn=None,
        obs_normalization=True):

        # Set default observations first
        # Define mins and maxes for each obseration
        self.obs_maxima = { 'focus': 200.,
            'speedX': 300., 'speedY': 300., 'speedZ': 300.,
            'angle': math.pi, 'damage':10000,
            'opponents': 200.,
            'rpm': 10000.,
            'track': 200.,
            'trackPos': 1.,
            'wheelSpinVel': 100.,
            "lap": 100,
            "img": 255
        }
        self.obs_minima = {'focus': 0.,
            'speedX': 0., 'speedY': 0., 'speedZ': 0.,
            'angle': 0., 'damage':0.,
            'opponents': 0.,
            'rpm': 0.,
            'track': 0.,
            'trackPos': 0.,
            'wheelSpinVel': 0.,
            "lap": 0,
            "img": 0
        }
        self.obs_dtypes = {
            'focus': np.float32,
            'speedX': np.float32,
            'speedY': np.float32,
            'speedZ': np.float32,
            'angle': np.float32,
            'damage': np.float32,
            'opponents': np.float32,
            'rpm': np.float32,
            'track': np.float32,
            'trackPos': np.float32,
            'wheelSpinVel': np.float32,
            "lap": np.uint8,
            "img": np.uint8
        }
        self.obs_dim = {'focus': 5,
            'speedX': 1, 'speedY': 1, 'speedZ': 1,
            'angle': 1, 'damage': 1,
            'opponents': 36,
            'rpm': 1,
            'track': 19,
            'trackPos': 1,
            'wheelSpinVel': 4,
            "lap": 1,
            "img": 1
            # "img": 64*64*3
        }

        if obs_vars is None:
            self.obs_vars = ['focus',
                'speedX', 'speedY', 'speedZ',
                'angle', 'damage',
                'opponents',
                'rpm',
                'track',
                'trackPos',
                'wheelSpinVel',
                "lap"]
            if vision:
                self.obs_vars.append( 'img')
        else:
            # TODO Add assertiong to check that the passed obs_vars are actually valid
            if (not vision and 'img' in obs_vars):
                print( "WARNING: Vision disabled but included in custon observation variables.")
                obs_vars.remove( 'img')
            if vision and  'img' not in obs_vars:
                pring( "WARNING: Vision enabled but not included in custom observation variables.")
                obs_vars.append( 'img')

            self.obs_vars = obs_vars
            # print( "Self obs vars:", self.obs_vars)

        # Set default observation preprocessing method
        self.obs_normalization = obs_normalization
        self.obs_preprocess_fn = obs_preprocess_fn

        # Set the default raceconfig file
        if race_config_path is None:
            race_config_path = os.path.join( os.path.dirname(os.path.realpath(__file__)),
                "raceconfigs/default.xml")

        #OpenAI Gym - Baselines and SubVecEnv compat fix
        self.seed_value = 42

        if len(self.obs_vars) == 1 and self.obs_vars[0] == "img":
            # VIsion only as observation
            self.observation_space = spaces.Box(low=0, high=255, shape=( 64, 64 ,3), dtype=np.uint8)
        else:
            high = np.hstack([ [self.obs_maxima[obs_name]] * self.obs_dim[obs_name] for obs_name in self.obs_vars])
            low = np.hstack([ [self.obs_minima[obs_name]] * self.obs_dim[obs_name] for obs_name in self.obs_vars])

            self.observation_space = spaces.Box( low=low, high=high, dtype=DEF_BOX_DTYPE)

        # Action spaces
        if throttle and gear_change:
            self.action_space = spaces.Box( low=np.array( [-1., -1., -1]),
                high=np.array( [1., 1., 6]),
                dtype=[np.float32, np.float32, np.int32])
        elif throttle:
            # Steeing and accel / decel
            self.action_space = spaces.Box( low=np.array( [-1., -1.]),
                high=np.array( [1., 1.]), dtype=np.float32)
        else:
            # Steering only
            self.action_space = spaces.Box( low=np.array( [-1.]),
                high=np.array( [1.]), dtype=np.float32)

        # Observation spaces
        # TODO: If time: Make this part configurable with obs list support and order Also
        # self.observation_space = spaces.Box( low)

        # Support for blackbox optimal reset
        self.reset_ep_count = 1
        self.hard_reset_interval = hard_reset_interval

        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.race_speed = race_speed
        self.rendering = rendering
        self.damage = damage
        self.recdata = recdata
        self.noisy = noisy

        # Treack randomization related
        self.randomisation = randomisation
        self.profile_reuse_count = 0
        self.profile_reuse_ep = profile_reuse_ep

        # Default
        self.initial_run = True

        # Raceconfig compat edit
        self.torcs_process_id = None
        self.race_config_path = race_config_path

        # Paralelization support
        self.rank = rank
        self.server_port = 3000 + self.rank
        # For one server instance, only one client supported
        # self.client_port = 3100 + self.rank*100

        # Freshly initialised
        if self.randomisation:
            self.randomise_track()

        # Internal time tracker for
        # The episode will end when the lap_limiter is reached
        # To put it simply if you want env to stap after 3 laps, set this to 4
        # Make sure to run torcs itself for more than 3 laps too, otherwise,
        # before terminating the episode
        self.lap_limiter = lap_limiter
        self.rec_episode_limit = rec_episode_limit
        self.rec_timestep_limit = rec_timestep_limit
        self.rec_index = rec_index

        ##print("launch torcs")
        #Just to be sure
        args = ["torcs", "-nofuel", "-nolaptime",
            "-a", str( self.race_speed)]

        if self.damage:
            args.append( "-nodamage")

        if self.noisy:
            args.append( "-noisy")

        if self.vision:
            args.append( "-vision")

        if not self.rendering:
            args.append( "-T") # Run in console

        if self.race_config_path is not None:
            args.append( "-raceconfig")
            # args.append( "\"" + race_config_path + "\"")
            args.append( self.race_config_path)

        if self.recdata:
            args.append( "-rechum %d" % self.rec_index)
            args.append( "-recepisodelim %d" % self.rec_episode_limit)
            args.append( "-rectimesteplim %d" % self.rec_timestep_limit)

        # For parallelization support
        args.append( "-p %d" % self.server_port)
        args.append("&")

        # print( "##### DEBUG: Args in init_torcs")
        # print( args)

        #Workaround: Sometimes the process has to be killed in them
        #SnakeOil3 file so we use the process_pid instead of the process object
        #my apologies
        self.torcs_process_id = subprocess.Popen( args, shell=False).pid

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """

    def randomise_track(self):
        # Desc: Randomizes the init positions of the bots, and luckily the agents
        # TODO: Randomize training tracks
        if self.profile_reuse_count == 0 or self.profile_reuse_count % self.profile_reuse_ep == 0:
            track_length = 2700 # Extract form torcs maybe
            max_pos_length = int(.7 * track_length) # Floor to 100 tile
            agent_init = random.randint(0,20) * 10
            bot_count = random.randint(1,10)
            min_bound = agent_init + 50
            max_leap = math.floor((max_pos_length - min_bound) / bot_count / 100) * 100
            bot_init_poss = []
            for _ in range(bot_count):
                bot_init_poss.append( random.randint( min_bound, min_bound + max_leap))
                # Random generate in range minbound and max pos length with max leap
                min_bound += max_leap

            # Check for random config file folder and create if not exists
            randconf_dir = os.path.join(  os.path.dirname(os.path.abspath(__file__)),
                "rand_raceconfigs")
            if not os.path.isdir(randconf_dir):
                os.mkdir(randconf_dir)
            randconf_filename = "agent_randfixed_%d" % agent_init
            for bot_idx in bot_init_poss:
                randconf_filename += "_%d" % bot_idx
            randconf_filename += ".xml"
            if not os.path.isfile( os.path.join( randconf_dir, randconf_filename)):
                # Create Fielk config based on xml template
                tree = None
                root = None
                with open( os.path.join( randconf_dir, "agent_randfixed_tmplt.xml")) as tmplt_f:
                    tree = ET.parse( tmplt_f)
                    root = tree.getroot()

                driver_node = None

                driver_section = root.find(".//section[@name='Drivers']")
                driver_section.append( ET.Element( "attnum",
                    { "name": "maximum_number", "val": "%d" % (1+ bot_count)}))
                driver_section.append( ET.Element( "attstr",
                    { "name": "focused module", "val": "scr_server" }))
                driver_section.append( ET.Element( "attnum",
                    { "name": "focused idx", "val": "1" }))

                # # Add Scr Server
                agent_section = ET.Element( "section",
                    { "name": "%d" % (1)})
                agent_section.append( ET.Element( "attnum",
                    { "name": "idx", "val": "%d" % (0) }))
                agent_section.append( ET.Element( "attstr",
                    { "name": "module", "val": "scr_server" }))
                driver_section.append( agent_section)

                driver_section.append( ET.Element( "attnum",
                    { "name": "initdist_%d" % (1), "val": "%d" % agent_init}))

                for bot_idx, bot_init_pos in enumerate( bot_init_poss):
                    bot_section = ET.Element( "section",
                        { "name": "%d" % (2+bot_idx)})
                    bot_section.append( ET.Element( "attnum",
                        { "name": "idx", "val": "%d" % (2+bot_idx) }))
                    bot_section.append( ET.Element( "attstr",
                        { "name": "module", "val": "fixed" }))
                    driver_section.append( bot_section)
                    driver_section.append( ET.Element( "attnum",
                        { "name": "initdist_%d" % (bot_idx+1), "val": "%d" % bot_init_pos}))

                randconf_savedir = "/tmp/randconf_dir_gymtorcs"
                if not os.path.isdir( randconf_savedir):
                    os.mkdir( randconf_savedir)
                randconf_abspath = os.path.join( randconf_savedir, randconf_filename)
                tree.write( randconf_abspath)

                self.race_config_path = randconf_abspath
                self.profile_reuse_count = 1

    def seed( self, seed_value=42):
        self.seed_value = seed_value

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs( u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        #track = np.array(obs['track'])
        #sp = np.array(obs['speedX'])
        #progress = sp*np.cos(obs['angle'])
        #reward = progress

        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        #progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        progress = sp*np.cos(obs['angle'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

        if int( obs["lap"]) > self.lap_limiter:
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0
        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True or self.reset_ep_count % self.hard_reset_interval == 0:
                self.reset_torcs()
                self.reset_ep_count = 1
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        ### dosssman: Pass existing process id and race config path
        if self.randomisation:
            self.randomise_track()

        self.client = snakeoil3.Client(p=self.server_port, vision=self.vision,
            process_id=self.torcs_process_id,
            race_config_path=self.race_config_path,
            race_speed=self.race_speed,
            rendering=self.rendering, lap_limiter=self.lap_limiter,
            damage=self.damage, recdata=self.recdata, noisy=self.noisy,
            rec_index = self.rec_index,rec_episode_limit=self.rec_episode_limit,
            rec_timestep_limit=self.rec_timestep_limit, rank=self.rank)  #Open new UDP in vtorcs

        self.client.MAX_STEPS = np.inf

        client = self.client

        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False

        # THe newly created TOrcs PID is also reattached to the Gym Torcs Env
        # This should be temporary ... but only time knows
        self.torcs_process_id = self.client.torcs_process_id

        self.reset_ep_count += 1
        self.profile_reuse_count += 1

        return self.get_obs()

    def end(self):
        # TODO:  Kill process by PID
        if self.torcs_process_id is not None:
            try:
                p = psutil.Process( self.torcs_process_id)
                #Kill children... yes
                for pchild in p.children():
                    pchild.terminate()
                #Then kill itself
                p.terminate()
            except Exception:
                self.torcs_process_id = None

    def close(self):
        self.end()

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        print( "Process PID: ", self.torcs_process_id)
        if self.torcs_process_id is not None:
            try:
                p = psutil.Process( self.torcs_process_id)
                #Kill children... yes
                for pchild in p.children():
                    pchild.terminate()
                #Then kill itself
                p.terminate()
            except Exception:
                ### TODO: Eventually FIGURE out what's woong
                #Hint:the process seems to already have beenkilled somewhereelse
                pass
            #Sad life to be a process

        if self.randomisation:
            self.randomise_track()

        args = ["torcs", "-nofuel", "-nolaptime",
            "-a", str( self.race_speed)]

        if self.damage:
            args.append( "-nodamage")

        if self.noisy:
            args.append( "-noisy")

        if self.vision:
            args.append( "-vision")

        if not self.rendering:
            args.append( "-T") # Run in console

        if self.race_config_path is not None:
            args.append( "-raceconfig")
            # args.append( "\"" + race_config_path + "\"")
            args.append( self.race_config_path)

        if self.recdata:
            args.append( "-rechum %d" % self.rec_index)
            args.append( "-recepisodelim %d" % self.rec_episode_limit)
            args.append( "-rectimesteplim %d" % self.rec_timestep_limit)

        args.append( "-p %d" % self.server_port)
        args.append("&")
        # print( "##### DEBUG: Args in reset_torcs")
        # print( args)
        self.torcs_process_id = subprocess.Popen( args, shell=False).pid

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        dict_obs = {}
        for obs_name in self.obs_vars:
            if obs_name == "img":
                image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])
                rsh = np.reshape( image_rgb, [64, 64, 3])
                dict_obs[obs_name] = rsh
            else:
                dict_obs[obs_name] = raw_obs[obs_name]
                if self.obs_normalization:
                    dict_obs[obs_name] = np.array(raw_obs[obs_name], dtype=self.obs_dtypes[obs_name])/(self.obs_maxima[obs_name] - self.obs_minima[obs_name])

        if callable( self.obs_preprocess_fn):
            return self.obs_preprocess_fn( dict_obs)

        return dict_obs
