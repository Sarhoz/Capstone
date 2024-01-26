## Capstone
Deep reinforcement learning for automated highway merging.


# Installation
Install the required packages using: `pip install -r requirements.txt`.

# Folders
The project contains a lot of different folders and code. Most of the code is not runnable and used to create the wanted files/videous/results/environment. The runnable code are in main.py and tuning.py. To understand how to use these codes a short descriptions of the different folders and codes need to be explained.
1. Functions (folder) contains three files with code: custom_hyperparams_opt.py, merge_in_env.py and utils.py.
custom_hyperparams_opt.py: contains the code used for tuning. This mainly has the possible hyperparameters that can/will be tuned.
merge_in_env.py: This is the created environment (for gymnasium) that results in the merging environment.
utils.py: This contains a lot of different codes. For example: functions that create/train/save models, register environments, performance and logger and many more things.
2. models (folder) contains all the results of the hyperparameter tuning of the different DRL models for the different environments.  These are made by code ran in tuning.py. This folder does not contain any code.
3. Tensorboard_log contains all the graphs of training all the different models in different environment.  These are made by code ran in main.py. This is used in Tensorboard environment. This folder does not contain any code.
4. Training models contains all the created models of the project. These models are used in the performance functions. These are made by code ran in main.py. This folder does not contain any code.
5. Video's.zip is a zip file that contains all the video's of the different models inside of different merge_in environments. These video are created by performance inside of main.py. This folder does not contain any code.


# Runnable Code
The project contains two runnable code files: main.py and tuning.py.
tuning.py: This code contains of changable variables at the start of the code. When the variables are as wanted/needed (for example N_TRAIN_ENVS AND N_EVAL_ENVS are equal to logical processors of your computer), than the code can be ran. Running the code could take a while (maybe even days). The result of this runnable code is put inside the folder called models. It also prints the hyperparameters of the best model.
main.py: This runnable code gives most of the results of this project. This code uses a lot of different functions from functions.utils. The code starts with making the environment. This is needed for all the other functions. Afterwards a lot of commented functions are given. These are used to create the different kind of models inside of the folder Training models. To use these codes, uncomment the function and let it run. This will take a while, because it needs time to train a model on a environment. The third function is called performance_model. This is the most important as it creates the videos and performances inside of performance.txt. The code contains a few inputs. To understand these look at the current performance_model code. By running this code another client will open that shows the car moving based on the created models. At the end, a graph will be given of all the different rewards on each run of car. By clicking on the red dot of the reward graph ---> the performances will be saved inside of Performance.txt. The last code is used to find the best hyperparameters made by tuning. This is used by putting the location of the file of the models hyperparameters found inside the models folder.

# Results
The results are shown in Performance.txt and the video's.zip.