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
        'type_learning': 'q_learning',
        'n_epochs': 20,
        'n_repeats': 3,
        # 'trajectory': False,
        # 'act_noise': 0.01
        # 'state_space': 'grdC',
    },
    'syst': {
        'n_homes': 1,
        'gan_generation': True,
    },
    'bat': {
        'SoC0': 1,
    }
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

    return log


log = find_previous_run_with_same_settings(settings, log_path)
print(f"runs: {log['run'].tolist()}")
if len(log) > 1:
    best_run = log['run'].loc[log['best_score_all'] == max(log['best_score_all'])].tolist()
    print(f"best run: {best_run}")
    print(f"best run with env-based exploration: "
          f"{log['run'].loc[log['best_score_env'] == max(log['best_score_env'])].tolist()}")