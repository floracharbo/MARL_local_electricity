from pathlib import Path
import pandas as pd
from src.post_analysis.log.organise_results import list_obs_to_str
from src.initialisation.input_data import get_settings_i, input_paths, input_params
from src.initialisation.initialise_objects import initialise_prm

results_analysis_path = Path("outputs/results_analysis")
log_path = results_analysis_path / "log_runs_from_pd.csv"
if not results_analysis_path.exists():
    print("no log")

candidates_left = True
settings = {
    'heat': {'file': 'heat2'},
    'RL': {
        'type_learning': 'facmac',
        'aggregate_actions': False,
        'state_space': [['avail_car_step', 'grdC', 'store_bool_flex']],
        'n_epochs': [20],
        'n_repeats': 3,
        'rnn_hidden_dim': [500],
        'evaluation_methods': [['env_r_c', 'opt']] * 1,
        'lr': 1e-3,
        'facmac': {'critic_lr': 5e-4},
        'ou_stop_episode': 1e3,  # for cqmix controller - training noise goes to zero after this episode
        'start_steps': [100],  # Number of steps for uniform-random action selection, before running real policy. Helps exploration.
        'hyper_initialization_nonzeros': 0.1,
    },
    'ntw': {
        'n': 10
    },
}
prm = input_paths()
settings_i = get_settings_i(settings, 0)
# initialise learning parameters, system parameters and recording
prm = input_params(prm, settings_i)
prm, profiles = initialise_prm(
    prm, 0, initialise_all=False
)

log = pd.read_csv(log_path)
for key, value in prm.items():
    for subkey, subvalue in value.items():
        if isinstance(subvalue, dict):
            for subsubkey, subsubvalue in subvalue.items():
                col_name = f"{subkey}_{subsubkey}"
                col_value = list_obs_to_str(subsubvalue) if col_name == 'state_space' else subsubvalue
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
                try:
                    if len(log.index[log[col_name] == col_value].tolist()) > 0:
                        log = log.loc[log[col_name] == col_value]
                    else:
                        print("no run found")
                        print(col_name)
                        candidates_left = False
                        break
                except Exception as ex:
                    print(ex)

if candidates_left:
    print(f"run(s) {log['run']}")
    print(log)
