from pathlib import Path

import pandas as pd

from src.initialisation.initialise_objects import initialise_prm
from src.initialisation.input_data import (get_settings_i, input_params,
                                           input_paths)
from src.post_analysis.log.organise_results import list_obs_to_str

results_analysis_path = Path("outputs/results_analysis")
log_path = results_analysis_path / "log_runs.csv"
if not results_analysis_path.exists():
    print("no log")

candidates_left = True
settings = {
    'heat': {'file': 'heat2'},
    'RL': {
        'type_learning': 'q_learning',
        # 'aggregate_actions': False,
        # 'state_space': [['avail_car_step', 'grdC', 'store_bool_flex']],
        'n_epochs': 200,
        # 'n_repeats': 3,
        # 'rnn_hidden_dim': [500],
        # 'evaluation_methods': [['env_r_c', 'opt']] * 1,
        # 'lr': 1e-3,
        # 'facmac': {'critic_lr': 5e-4},
        # 'ou_stop_episode': 1e3,
        # 'start_steps': [100],
        # 'hyper_initialization_nonzeros': 0.1,
    },
    'syst': {
        'n_homes': 10
    },
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
        print(f"run(s) {log['run']}")
        print(log)

    return log


log = find_previous_run_with_same_settings(settings, log_path)
if len(log) > 1:
    print(f"best run: {log.loc[log['best_score'] == max(log['best_score'])]}")
    print(f"best run with env-based exploration: "
          f"{log.loc[log['best_score_env'] == max(log['best_score_env'])]}")
