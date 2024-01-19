from Dennis.custom_hyperparams_opt import HYPERPARAMS_SAMPLER
from Dennis.utils import sequential_dir

import joblib
import os
from typing import Any, Dict

import gymnasium as gym
from highway_env_copy.envs.merge_in_env import *
# gym.register(id="merge-in-v0", entry_point="merge_in_env:MergeinEnv")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sb3_contrib import TRPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn


EXPANSIVE = False
N_TRIALS = 100
N_JOBS = 1
N_THREADS = -1
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 15)
PROGRESS = {
    "optimize": True,
    "learn": True
    }

ALGO = PPO
ROOT_PATH = "models/" + ALGO.__name__ + "/tuning"
ALGO_PATH = sequential_dir(ROOT_PATH, return_path=True)
ENV_ID = "merge-in-v0"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
    }
SAMPLED_HYPERPARAMS = HYPERPARAMS_SAMPLER[ALGO.__name__.lower()]


class TrialEvalCallback(EvalCallback):
    
    """
    Callback used for evaluating and reporting a trial.
    
    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0
        ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose
            )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(SAMPLED_HYPERPARAMS(trial, EXPANSIVE))
    # Create the RL model.
    model = ALGO(**kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(gym.make(ENV_ID))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ)

    nan_encountered = False
    try:
        # Train the model
        if PROGRESS["learn"]:
            model.learn(N_TIMESTEPS, callback=[eval_callback, ProgressBarCallback()])
        else:
            model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
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
    try:
        if PROGRESS["optimize"]:
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, show_progress_bar=True)
        else:
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
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


if __name__ == "__main__":
    hyperparameter_tuning()
