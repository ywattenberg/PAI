import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
import optuna
from solution import Agent
from multiprocessing import Pool

def objective(trial):
    gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
    tau = trial.suggest_uniform('tau', 0.0, 1.0)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    lr_step_size = trial.suggest_int('lr_step_size', 100, 1000)
    lr_gamma = trial.suggest_uniform('lr_gamma', 0.0, 1.0)

    TRAIN_EPISODES = 50
    TEST_EPISODES = 50
    agent = Agent(gamma=gamma, tau=tau, lr=lr, lr_step_size=lr_step_size, lr_gamma=lr_gamma)
    env = get_env(g=10.0, train=True)
    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, False, train=True)

    env = get_env(g=10.0, train=False)
    test_returns = []
    for EP in range(TEST_EPISODES):
        rec = None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, False, train=False)
        test_returns.append(episode_return)
    
    avg_test_return = np.mean(np.array(test_returns))
    return avg_test_return

def run_study(worker_id):
    print(f"Worker {worker_id} started")
    study = optuna.load_study(study_name='test', storage='sqlite:///test.db')
    study.optimize(objective, n_trials=500)


if __name__ == "__main__":
    with Pool(16) as p:
        p.map(run_study, range(16))
    study = optuna.load_study(study_name='test', storage='sqlite:///test.db')
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)


