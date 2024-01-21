from Dennis.custom_hyperparams_opt import HYPERPARAMS_SAMPLER
from Dennis.utils import sequential_dir, round_to_mult

import joblib
import os
from typing import Any, Dict
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
#from highway_env_copy.envs.merge_in_env import *
#gym.register(id="merge-in-v0", entry_point="merge_in_env:MergeinEnv")

import optuna
# from optuna.integration.tensorboard import TensorBoardCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from sb3_contrib import TRPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.utils import ALGOS

import torch
import torch.nn as nn



EXPANSIVE = False
COUNTED_STATES = [TrialState.COMPLETE, TrialState.RUNNING, TrialState.PRUNED]
N_TRIALS = 100
N_JOBS = 1      # way slower training when > 1
N_THREADS = 1
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 10
N_TRAIN_ENVS = 12
N_EVAL_ENVS = 12
MP_CLS = SubprocVecEnv if N_TRAIN_ENVS > 1 else DummyVecEnv
TIMEOUT = None     # int(60 * 15)
PROGRESS = {
    "optimize": True,
    "learn": False
    }

ALGO = "dqn"
ENV_ID = "merge-in-v0"
ROOT_PATH = "models/" + ALGO.upper() + "/merge_in_" + ENV_ID.split("-")[2] + "/tuning"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
    }
SAMPLED_HYPERPARAMS = HYPERPARAMS_SAMPLER[ALGO.lower()]



def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs["env"] = make_vec_env(ENV_ID, n_envs=N_TRAIN_ENVS, vec_env_cls=MP_CLS) if N_TRAIN_ENVS > 1 else ENV_ID
    # Sample hyperparameters.
    kwargs.update(SAMPLED_HYPERPARAMS(trial, EXPANSIVE))
    # Create the RL model.
    model = ALGOS[ALGO](**kwargs)
    # Create env used for evaluation.
    eval_env = make_vec_env(ENV_ID, n_envs=N_EVAL_ENVS, vec_env_cls=MP_CLS)# if N_EVAL_ENVS > 1 else Monitor(gym.make(ENV_ID))
    # Create the callbacks that will periodically evaluate and report the performance.
    pbar_callback = ProgressBarCallback()
    eval_callback = TrialEvalCallback(eval_env=eval_env, trial=trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=max(EVAL_FREQ // N_EVAL_ENVS, 1), best_model_save_path=ALGO_PATH+"/best_model")
    callbacks = CallbackList([pbar_callback, eval_callback]) if PROGRESS["learn"] else eval_callback

    nan_encountered = False
    try:
        # Train the model
        model.learn(total_timesteps=N_TIMESTEPS, callback=callbacks)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Clean progress bar
        if PROGRESS["learn"]:
            pbar_callback.on_training_end()
        # Free memory.
        assert model.env is not None
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def hyperparameter_tuning():
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(N_THREADS)
    # Select the sampler
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS//3)
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    # Track trials on tensorboard and enforce limit on trials via callbacks
    # tb_callback = TensorBoardCallback(ALGO_PATH+"/logs", metric_name="mean reward")
    num_trials_callback = MaxTrialsCallback(n_trials=N_TRIALS, states=COUNTED_STATES)
    # Reduce logging when using progress bars
    if any([*PROGRESS.values()]):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        # Note: we count already running trials here otherwise we get
        #  (n_trials + number of workers) trials in total.
        if len(study.get_trials(states=COUNTED_STATES)) < N_TRIALS:
            study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, n_jobs=N_JOBS, callbacks=[num_trials_callback], show_progress_bar=PROGRESS["optimize"])  #[tb_callback, num_trials_callback]
    except KeyboardInterrupt:   # !!!!! DOES NOT WORK WHEN N_ENVS IS NOT 0 !!!!!
        # Save Study
        joblib.dump(study, ALGO_PATH+"/study_interrupted.pkl")
    else:
        joblib.dump(study, ALGO_PATH + "/study.pkl")

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


    # Show and save results
        study.trials_dataframe().to_csv(ALGO_PATH + "/study_results.csv")

        # try:
        #     fig1 = plot_optimization_history(study)
        #     fig2 = plot_param_importances(study)

        #     fig1.show()
        #     fig2.show()
        # except (ValueError, ImportError, RuntimeError):
        #     pass


if __name__ == "__main__":
    ALGO_PATH = sequential_dir(ROOT_PATH, return_path=True)
    N_TIMESTEPS = round_to_mult(N_TIMESTEPS, 2048*N_TRAIN_ENVS)
    hyperparameter_tuning()
