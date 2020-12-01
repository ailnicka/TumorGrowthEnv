from gym.envs.registration import register

register(
    id='TumorGrowthEnv-v0',
    entry_point='TumorGrowthEnv.envs:TumorGrowthEnv',
)