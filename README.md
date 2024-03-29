[![Makefile CI](https://github.com/floracharbo/private_MARL/actions/workflows/makefile.yml/badge.svg)](https://github.com/floracharbo/private_MARL/actions/workflows/makefile.yml)

# Multi-agent reinforcement learning for residential energy flexibility

Contact: Flora Charbonnier, flora.charbonnier@eng.ox.ac.uk
## What is it?

This project uses multi-agent reinforcement learning for the coordination of residential energy flexibility.

Cooperating agents learn to control the flexibility offered by electric vehicles, space heating and flexible loads in a partially observable stochastic environment.

The novel combination of learning from off-line convex optimisations on historical data and isolating marginal contributions to total rewards in reward signals increases stability and performance at scale. This tackles the challenge of scalability for simultaneously learning independent agents under partial observability in a stochastic environment for household-level decision-making.

The houses (agents) cooperate to minimise global system costs consisting of grid, distribution and storage costs.

For more details on the system framework, a description can be found in the peer-reviewed paper below, which uses this repository code:

Charbonnier F, Morstyn T, McCulloch MD. Scalable multi-agent reinforcement learning for distributed control of residential energy flexibility. Appl Energy 2022;314:118825. https://doi.org/10.1016/j.apenergy.2022.118825.

## Main Features

The learning methods are:
- independent learners: Q-learning, DDPG, DQN, DDQN
- FACMAC (https://papers.nips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html)

The flexible technologies modelled are:
- electric vehicles (including realistic random training and evaluation consumption and availability profiles)
- heat pumps (modelled based on required comfort temperature profiles, house modelling and external temperatures)
- generic flexible household loads (realistic random training and evaluation profiles generated)

Exploration sources are:
- Environment exploration. Traditionally, agents collect experience by interacting with an environment. 
- Optimisations. A novel approach collects experience from optimisations. Agents learn from convex “omniscient” optimisations on historical data with perfect knowledge of current and future variables. This experience is then used under partial observability and control for stable coordination between prosumers at scale. A standard convex optimiser uses the same data that would be used to populate the environment explorations but solves over the whole day-horizon with perfect knowledge of all variables. Then, at each time step, the system variables are translated into equivalent RL 
 tuples for each agent, which are used to update the policies in the same way as for standard Q-learning as presented below.

The learning for independent learners can be
- Global value-function / Q-table: this does not scale beyond 3 agents for Q-learning.
- Centralised learning. A single home-level Q-table uses experience from all agents during pre-learning. All agents use the centrally learned policy for decentralised implementation.
- Distributed learning. Each agent learns its table with its own experience. No information is shared between agents.

The action space can be
- aggregated into a single action. This reduces the state space for implementation with Q-learning.
- disaggregated into the heating, EV charging, and household load flexibility action variables. This yields better performance using FACMAC.

Rewards for independent learners can be defined as:
- Total reward. The instantaneous total system reward is used to update the Q-table for all agents.
- Marginal reward. The difference in total instant rewards between that if agent 
 selects the greedy action and that if it selects the default action is used
- Advantage reward. The post difference between values when performs the greedy and the default action is used. 
- Count. The Q-table stores the number of times each state–action pair is selected by the optimiser.

Energy prices follow historical Octopus's agile tariff.

## Where to get it
The source code for **MARL_local_electricity** is currently hosted on GitHub at: https://github.com/floracharbo/MARL_local_electricity

The data pre-processing for the home energy data generation **HEDGE** can be found at: https://github.com/floracharbo/HEDGE.

## Usage
1. Obtain input data for **HEDGE**, as intructed on the README in https://github.com/floracharbo/HEDGE.

2. Create virtual environment: 
```sh
conda create -n "my_venv" python==3.9.13
```

2. Install packages, inserting your operating system (os) in the yml file name below (mac or linux).
```sh
conda env update --name my_venv --file config_files/environments/environment_[os].yml --prune
```

3. Obtain Mosek licence and move to adequate folder.
e.g. https://www.mosek.com/products/academic-licenses/

4. Activate environment:
```sh
conda activate my_venv
```

5. Define user inputs
Default settings are in the inputs folder `config_files/input_parameters`:
- car.yaml: battery inputs
- gen.yaml: PV generation inputs
- heat.yaml: heating inputs
- loads.yaml: household loads inputs
- grd.yaml: electricity grid inputs
- paths.yaml: user paths inputs
- RL.yaml: reinforcement learning inputs
- save.yaml: data savings inputs
- syst.yaml: general system inputs


If the user wishes to change these default settings, whilst still being able to compare previous runs in `organise_results`, the previous default settings should be recorded in `input_parameters/previous_defaults.yaml`.

To use experiment-specific settings different to the default parameters in `config_files/default_input_parameters`, enter specific values in the `settings` dictionary in the `main.py` file, as shown in as an example in the file.
If values entered are in a list format, multiple experiments will loop though these values, so multiple experiments can be planned in one run.


6. Run`main.py`

Alternatively, settings can be entered through the command line rather than from the .py or .yaml files, e.g.

`python main.py -n 5 --n_repeats 2 --n_epochs 20 -o grdC --o avail_car_step --rnn_hidden_dim 100`

Where `-o` stands for observation, `-n` for the number of homes. You can add more abbreviations in the `get_settings_i` function in the `initialise_prm.py` file.

7. Obtain results in`results` folder.

## License
[GNU AGPLv3](LICENSE.txt)

## Open-source data references
- UK electricity grid carbon intensity from https://data.nationalgrideso.com/carbon-intensity1/historic-generation-mix
- UK electriciy grid prices from Octopus Agile Tariff, Oxford, "https://api.octopus.energy"
- outdoors temperature data from the NASA MERRA 2 inst1_2d_asm_Nx (M2I1NXASM) reanalysis dataset, Oxford. https://disc.gsfc.nasa.gov/datasets/M2I1NXASM_5.12.4/summary?keywords=M2I1NXASM&start=1920-01-01&end=2022-10-26
