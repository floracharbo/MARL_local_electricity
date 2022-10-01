# adapted from
# https://github.com/oxwhirl/facmac

from learners.facmac.modules.agents.comix_agent import (CEMAgent,
                                                        CEMRecurrentAgent)
from learners.facmac.modules.agents.mlp_agent import MLPAgent
from learners.facmac.modules.agents.qmix_agent import FFAgent, QMIXRNNAgent
from learners.facmac.modules.agents.rnn_agent import RNNAgent

REGISTRY = {}

REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
