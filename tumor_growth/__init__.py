from gym.envs.registration import register

register(
    id='tumor_growth-v0',
    entry_point='tumor_growth.envs:TumorGrowthEnv',
)