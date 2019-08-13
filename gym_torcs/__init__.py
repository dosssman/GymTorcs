from gym.envs.registration import register

# Torcs Custom
register(
	id="Torcs-v0",
	entry_point='gym_torcs.torcs_env:TorcsEnv'
)
