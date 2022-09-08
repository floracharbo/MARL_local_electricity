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
- Q-learning
- DDPG
- DQN
- DDQN
- FACMAC (https://papers.nips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html)

The flexible technologies modelled are:
- electric vehicles (including realistic random training and evaluation consumption and availability profiles)
- heat pumps (modelled based on required comfort temperature profiles, house modelling and external temperatures)
- generic flexibile household loads (realistic random training and evaluation profiles generated)

Exploration sources are:
- Environment exploration. Traditionally, agents collect experience by interacting with an environment. 
- Optimisations. A novel approach collects experience from optimisations. Agents learn from convex “omniscient” optimisations on historical data with perfect knowledge of current and future variables. This experience is then used under partial observability and control for stable coordination between prosumers at scale. A standard convex optimiser uses the same data that would be used to populate the environment explorations but solves over the whole day-horizon with perfect knowledge of all variables. Then, at each time step, the system variables are translated into equivalent RL 
 tuples for each agent, which are used to update the policies in the same way as for standard Q-learning as presented below.

The learning can be
- global value-function / Q-table: this does not scale beyond 3 agents for Q-learning.
- Centralised learning. A single home-level Q-table uses experience from all agents during pre-learning. All agents use the centrally learned policy for decentralised implementation.
- Distributed learning. Each agent learns its table with its own experience. No information is shared between agents.

The action space can be
- aggregated into a single action. This reduces the state space for implementation with Q-learning.
- disaggregated into the heating, EV charging, and household load flexibility action variables. This yields better performance using FACMAC.

Rewards can be defined as:
- Total reward. The instantaneous total system reward is used to update the Q-table for all agents.
- Marginal reward. The difference in total instant rewards between that if agent 
 selects the greedy action and that if it selects the default action is used
- Advantage reward. The post difference between 
 values when performs the greedy and the default action is used. 
- Count. The Q-table stores the number of times each state–action pair is selected by the optimiser.

Energy prices follow historical Octopus's agile tariff.

## Where to get it
The source code for **MARL_local_electricity** is currently hosted on GitHub at: https://github.com/floracharbo/MARL_local_electricity

The data pre-processing for the home energy data generation **HEDGE** can be found at: https://github.com/floracharbo/HEDGE.

## Usage
1. Create virtual environment: 
```sh
python -m venv my_venv
```
3. Activate virtual environment: 
```sh
source my_venv/bin/activate
```
4. Install requirements: 
```sh
make install
```

5. Define user inputs
User inputs are in the inputs folder
- bat.yaml: battery inputs
- gen.yaml: PV generation inputs
- grd.yaml: electricity grid inputs
- heat2.yaml: heating putings
- loads.yaml: household loads inputs
- ntw.yaml: network inputs
- paths.yaml: user paths inputs
- RL.yaml: reinforcement learning inputs
- save.yaml: data savings inputs
- syst.yaml: general system inputs

The user can modify entries in the yaml files, or keep these as default values and enter specific values in the main_rl.py file in the settings dictionary, as shown in as an example in the file.
If values entered are in a list format, multiple experiments will loop though these values, so multiple experiments can be planned in one run.

6. Run`main_rl.py`
7. Obtain results in`results` folder.
## License
[GNU AGPLv3](LICENSE.txt)
