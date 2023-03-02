# from https://github.com/oxwhirl/facmac

import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs,
                            n_agents, gamma, td_lambda):
    # Assumes <target_qs > in B*T*A and <reward >, <terminated >,
    # <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
            * (rewards[:, t] + (1 - td_lambda) * gamma
                * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def preprocess_scheme(scheme, preprocess):

    if preprocess is not None:
        for k in preprocess:
            assert k in scheme
            new_k = preprocess[k][0]
            transforms = preprocess[k][1]

            vshape = scheme[k]["vshape"]
            dtype = scheme[k]["dtype"]
            for transform in transforms:
                vshape, dtype = transform.infer_output_info(vshape, dtype)

            scheme[new_k] = {
                "vshape": vshape,
                "dtype": dtype
            }
            if "group" in scheme[k]:
                scheme[new_k]["group"] = scheme[k]["group"]
            if "episode_const" in scheme[k]:
                scheme[new_k]["episode_const"] = scheme[k]["episode_const"]

    return scheme


def input_last_action(obs_last_action, inputs, batch, t):
    if obs_last_action:
        if t == 0:
            inputs.append(th.zeros_like(batch["actions"][:, t]))
        else:
            inputs.append(batch["actions"][:, t - 1])

    return inputs
