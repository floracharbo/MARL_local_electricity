from pathlib import Path

import pandas as pd

from src.environment.initialisation.initialise_objects import initialise_prm
from src.environment.initialisation.input_data import (get_settings_i,
                                                       input_params,
                                                       input_paths)
from src.environment.post_analysis.analyse_runs.organise_results import \
    list_obs_to_str

results_analysis_path = Path("outputs/results_analysis")
log_path = results_analysis_path / "log_runs.csv"
if not results_analysis_path.exists():
    print("no log")

candidates_left = True
settings = {
    'RL': {
        'type_learning': 'facmac',
        'n_epochs': 20,
        'n_repeats': 10,
        # 'n_end_test': 30,
        'trajectory': False,
        'state_space': 'grdC_min_voltage',
        # 'supervised_loss': False,
        # 'state_space': 'grdC',
        # 'buffer_size': 5000,
        # 'act_noise': 0,
        # 'facmac': {'critic_lr': 1e-2},
        'reactive_power_for_voltage_control': True,
        # 'lr': 1e-4,
        # 'q_learning-eps': 0.1,
        # 'n_discrete_actions': 10,
        # 'act_noise': 0.01
        # 'state_space': 'grdC',
    },
    'syst': {
        'n_homes': 55,
        'gan_generation': True,
    },
    'bat': {
        'SoC0': 1,
    },
    'grd': {'penalty_overvoltage': 1e2},
    # 'grd': {'simulate_panda_power_only': True},
}
SETTINGS_ONLY = True


def initialise_settings_prm_log(settings, log_path, settings_only=True):
    if not settings_only:
        prm = input_paths()
    settings_i = get_settings_i(settings, 0)
    # initialise learning parameters, system parameters and recording
    if settings_only:
        prm = settings_i
    else:
        prm = input_params(prm, settings_i)
        prm = initialise_prm(
            prm, 0, initialise_all=False
        )

    log = pd.read_csv(log_path)

    return prm, log


def find_previous_run_with_same_settings(settings, log_path):
    prm, log = initialise_settings_prm_log(settings, log_path, settings_only=SETTINGS_ONLY)
    candidates_left = True
    for key, value in settings.items():
        for subkey, subvalue in value.items():
            if isinstance(subvalue, dict):
                for subsubkey, subsubvalue in subvalue.items():
                    col_name = f"{subkey}_{subsubkey}"
                    col_value = list_obs_to_str(subsubvalue) if col_name == 'state_space' \
                        else subsubvalue
                    if col_name in log.columns:
                        if len(log.index[log[col_name] == col_value].tolist()) > 0:
                            log = log.loc[log[col_name] == col_value]
                        else:
                            print("no run found")
                            print(col_name)
                            candidates_left = False
                            break
            else:
                col_name = subkey
                col_value = list_obs_to_str(subvalue) if col_name == 'state_space' else subvalue
                if col_name in log.columns:
                    if len(log.index[log[col_name] == col_value].tolist()) > 0:
                        log = log.loc[log[col_name] == col_value]
                    else:
                        print("no run found")
                        print(col_name)
                        candidates_left = False
                        break

    if candidates_left:
        print(f"run(s) {log['run'].tolist()}")
        print(log[['run', 'best_score_all']])

    return log, candidates_left


log, candidates_left = find_previous_run_with_same_settings(settings, log_path)

if candidates_left:
    print(f"runs: {log['run'].tolist()}")
    if len(log) > 1:
        best_run = log['run'].loc[log['best_score_all'] == max(log['best_score_all'])].tolist()
        print(f"best run: {best_run}")
        print(f"best run with env-based exploration: "
              f"{log['run'].loc[log['best_score_env'] == max(log['best_score_env'])].tolist()}")
