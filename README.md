## Tumor Growth Open AI Gyn Environment
The [Open AI Gym](https://gym.openai.com) environment based on the 
[EMT6/Ro multicellular tumor spheroids simulation](https://github.com/banasraf/EMT6-Ro/).

To be able to use it, the wheel for the above simulator needs to be pip installed, 
and then this repository needs to be pip installed.
After it, the environment can be used from Gym with:
 `gym.make('tumor_growth:tumor_growth_env-v0')`.