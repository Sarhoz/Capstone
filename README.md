## Capstone
Deep reinforcement learning for automated highway merging


## Installation
First install the required packages using: `pip install -r requirements.txt`.
Afterwards delete the highway-env on the venv (a library that has been installed) inside the site-packages. The highway_env_copy should than be coppied and placed inside the venv site-packages and be renamed to "highway_env".

## Main codes used
1. Main.py is used only to call the function.py. The function.py codes all the code for creating the results of the project. It contains the models and performance.
2. Highway_env_copy is the library that has been used to create the environment. The main codes that have been changed are called "merge_in_env.py" inside of envs and "Kinematics.py" inisde of vehicle.
3. The hyperparameter tuning has been done inside of the code called "tuning.py". This code is only used for tuning of hyperparameters for models inside of environments.

## Results
The results of the models have been stored inside of different folders:
Tensorboard_log: contains all the graphs of training all the different models
Training models: contains all the created models of the project
models: contains all the results of the hyperparameter tuning of the different DRL models for the different environments.
Performance.txt: contains the performance of the different models!
Video's: contains the video recording of the different DRL models in the different environments.