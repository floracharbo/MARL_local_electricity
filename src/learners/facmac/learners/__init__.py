# adapted from
# https://github.com/oxwhirl/facmac

from src.learners.facmac.learners.facmac_learner import FACMACLearner
from src.learners.facmac.learners.facmac_learner_discrete import \
    FACMACDiscreteLearner

from .cq_learner import CQLearner
from .maddpg_learner import MADDPGLearner

# from .maddpg_learner_discrete import MADDPGDiscreteLearner

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
# REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
