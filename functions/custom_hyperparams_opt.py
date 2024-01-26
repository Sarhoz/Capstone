from typing import Any, Dict

import numpy as np
import optuna
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

from rl_zoo3 import linear_schedule


def sample_ppo_params(trial: optuna.Trial, expansive:bool=False) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    if expansive:
        # Model parameters
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])    # default = 64
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])    # default = 2048
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])    # default = 0.99
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)    # default = 0.0003 (3e-4)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)    # default = 0.0
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])    # default = 0.2
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])    # default = 10
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])    # default = 0.95
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])    # default = 0.5
        vf_coef = trial.suggest_float("vf_coef", 0, 1)    # default = 0.5
        # Uncomment for gSDE (continuous action)
        # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])    # default = -1


        # Policy parameters
        net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])    # default = small
        # Uncomment for gSDE (continuous actions)
        # log_std_init = trial.suggest_float("log_std_init", -4, 1)    # default = 0.0
        
        # Orthogonal initialization
        ortho_init = False
        # ortho_init = trial.suggest_categorical('ortho_init', [False, True])    # default = True
        
        # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])    # default = tanh
        
        # lr_schedule = "constant"
        # Uncomment to enable learning rate schedule
        # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        # if lr_schedule == "linear":
        #     learning_rate = linear_schedule(learning_rate)

        # TODO: account when using multiple envs
        if batch_size > n_steps:
            batch_size = n_steps

        # Independent networks usually work best
        # when not working with images
        net_arch = {
            "tiny": dict(pi=[64], vf=[64]),
            "small": dict(pi=[64, 64], vf=[64, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256]),
        }[net_arch_type]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]
        
        # print(f"Tuning expansively")
        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            # "sde_sample_freq": sde_sample_freq,
            "policy_kwargs": dict(
                # log_std_init=log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        }
    else:
        # Model parameters
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])    # default = 64
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])    # default = 0.99
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.01, log=True)    # default = 0.0003 (3e-4)
        
        # print(f"Limited tuning")
        return {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
        }


def sample_trpo_params(trial: optuna.Trial, expansive:bool=False) -> Dict[str, Any]:
    """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
    if expansive:
        # Model parameters
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])    # default = 128
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])    # default = 2048
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])    # default = 0.99
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)    # default = 0.001 (1e-3)
        # line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])    # default = 0.8
        n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])    # default = 10
        cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])    # default = 15
        # cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])    # default = 0.1
        target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])    # default = 0.01
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])    # default = 0.95
        # Uncomment for gSDE (continuous action)
        # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])    # default = -1

        
        # Policy parameters
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])    # default = small
        # Uncomment for gSDE (continuous actions)
        # log_std_init = trial.suggest_float("log_std_init", -4, 1)    # default = 0.0
        
        # Orthogonal initialization
        ortho_init = False
        # ortho_init = trial.suggest_categorical('ortho_init', [False, True])    # default = True
        
        # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])    # default = tanh
        
        # lr_schedule = "constant"
        # Uncomment to enable learning rate schedule
        # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        # if lr_schedule == "linear":
        #     learning_rate = linear_schedule(learning_rate)

        # TODO: account when using multiple envs
        if batch_size > n_steps:
            batch_size = n_steps

        # Independent networks usually work best
        # when not working with images
        net_arch = {
            "small": dict(pi=[64, 64], vf=[64, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256]),
        }[net_arch_type]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

        # print(f"Tuning expansively")
        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            # "cg_damping": cg_damping,
            "cg_max_steps": cg_max_steps,
            # "line_search_shrinking_factor": line_search_shrinking_factor,
            "n_critic_updates": n_critic_updates,
            "target_kl": target_kl,
            "learning_rate": learning_rate,
            "gae_lambda": gae_lambda,
            # "sde_sample_freq": sde_sample_freq,
            "policy_kwargs": dict(
                # log_std_init=log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        }
    else:
        # Model parameters
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])    # default = 64
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])    # default = 0.99
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.01, log=True)    # default = 0.001 (1e-3)
        
        # print(f"Limited tuning")
        return {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
        }


def sample_dqn_params(trial: optuna.Trial, expansive:bool=False) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    if expansive:
        # Model parameters
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])    # default = 0.99
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)    # default = 0.0001 (1e-4)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])    # default = 32
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])    # default = 1000000 (1e6)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)    # default = 1.0
        exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)    # default = 0.1
        target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])    # default = 10000 (1e4)
        learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])    # default = 100
        
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])    # default = 4
        subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
        gradient_steps = max(train_freq // subsample_steps, 1)    # default = 1


        # Policy parameters
        net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])    # default = small

        net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch_type]

        # print(f"Tuning expansively")
        return {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "target_update_interval": target_update_interval,
            "learning_starts": learning_starts,
            "policy_kwargs": dict(net_arch=net_arch),
        }
    else:
        # Model parameters
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])    # default = 0.99
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.01, log=True)    # default = 0.0001
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])    # default = 32

        # print(f"Limited tuning")
        return {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
        }


HYPERPARAMS_SAMPLER = {
    "dqn": sample_dqn_params,
    "ppo": sample_ppo_params,
    "trpo": sample_trpo_params,
}
