from gym.envs.registration import register

register(
    id='tumor_growth_env-v0',
    entry_point='tumor_growth_env.envs:tumor_growth_env',
)