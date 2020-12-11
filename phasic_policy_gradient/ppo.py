"""
Mostly copied from ppo.py but with some extra options added that are relevant to phasic
"""

import numpy as np
import torch as th
from mpi4py import MPI
from .tree_util import tree_map
from . import torch_util as tu
from .log_save_helper import LogSaveHelper
from .minibatch_optimize import minibatch_optimize
from .roller import Roller
from .reward_normalizer import RewardNormalizer

import math
from . import logger

INPUT_KEYS = {"ob", "ac", "first", "logp", "vtarg", "adv", "state_in"}

def compute_gae(
    *,
    vpred: "(th.Tensor[1, float]) value predictions",
    reward: "(th.Tensor[1, float]) rewards",
    first: "(th.Tensor[1, bool]) mark beginning of episodes",
    γ: "(float)",
    λ: "(float)"
):
    orig_device = vpred.device
    assert orig_device == reward.device == first.device
    vpred, reward, first = (x.cpu() for x in (vpred, reward, first))
    first = first.to(dtype=th.float32)
    assert first.dim() == 2
    nenv, nstep = reward.shape
    assert vpred.shape == first.shape == (nenv, nstep + 1)
    adv = th.zeros(nenv, nstep, dtype=th.float32)
    lastgaelam = 0
    for t in reversed(range(nstep)):
        notlast = 1.0 - first[:, t + 1]
        nextvalue = vpred[:, t + 1]
        # notlast: whether next timestep is from the same episode
        delta = reward[:, t] + notlast * γ * nextvalue - vpred[:, t]
        adv[:, t] = lastgaelam = delta + notlast * γ * λ * lastgaelam
    vtarg = vpred[:, :-1] + adv
    return adv.to(device=orig_device), vtarg.to(device=orig_device)

def log_vf_stats(comm, **kwargs):
    logger.logkv(
        "VFStats/EV", tu.explained_variance(kwargs["vpred"], kwargs["vtarg"], comm)
    )
    for key in ["vpred", "vtarg", "adv"]:
        logger.logkv_mean(f"VFStats/{key.capitalize()}Mean", kwargs[key].mean())
        logger.logkv_mean(f"VFStats/{key.capitalize()}Std", kwargs[key].std())

def compute_advantage(model, seg, γ, λ, comm=None):
    comm = comm or MPI.COMM_WORLD
    finalob, finalfirst = seg["finalob"], seg["finalfirst"]
    vpredfinal = model.v(finalob, finalfirst, seg["finalstate"])
    reward = seg["reward"]
    logger.logkv("Misc/FrameRewMean", reward.mean())
    adv, vtarg = compute_gae(
        γ=γ,
        λ=λ,
        reward=reward,
        vpred=th.cat([seg["vpred"], vpredfinal[:, None]], dim=1),
        first=th.cat([seg["first"], finalfirst[:, None]], dim=1),
    )
    log_vf_stats(comm, adv=adv, vtarg=vtarg, vpred=seg["vpred"])
    seg["vtarg"] = vtarg
    adv_mean, adv_var = tu.mpi_moments(comm, adv)
    seg["adv"] = (adv - adv_mean) / (math.sqrt(adv_var) + 1e-8)

def compute_losses(
    model,
    ob,
    ac,
    first,
    logp,
    vtarg,
    adv,
    state_in,
    clip_param,
    vfcoef,
    entcoef,
    kl_penalty,
):
    losses = {}
    diags = {}
    pd, vpred, aux, _state_out = model(ob=ob, first=first, state_in=state_in)
    newlogp = tu.sum_nonbatch(pd.log_prob(ac))
    # prob ratio for KL / clipping based on a (possibly) recomputed logp
    logratio = newlogp - logp
    ratio = th.exp(logratio)

    if clip_param > 0:
        pg_losses = -adv * ratio
        pg_losses2 = -adv * th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pg_losses = th.max(pg_losses, pg_losses2)
    else:
        pg_losses = -adv * th.exp(newlogp - logp)

    diags["entropy"] = entropy = tu.sum_nonbatch(pd.entropy()).mean()
    diags["negent"] = -entropy * entcoef
    diags["pg"] = pg_losses.mean()
    diags["pi_kl"] = kl_penalty * 0.5 * (logratio ** 2).mean()

    losses["pi"] = diags["negent"] + diags["pg"] + diags["pi_kl"]
    losses["vf"] = vfcoef * ((vpred - vtarg) ** 2).mean()

    with th.no_grad():
        diags["clipfrac"] = (th.abs(ratio - 1) > clip_param).float().mean()
        diags["approxkl"] = 0.5 * (logratio ** 2).mean()

    return losses, diags

def learn(
    *,
    venv: "(VecEnv) vectorized environment",
    model: "(ppo.PpoModel)",
    interacts_total: "(float) total timesteps of interaction" = float("inf"),
    nstep: "(int) number of serial timesteps" = 256,
    γ: "(float) discount" = 0.99,
    λ: "(float) GAE parameter" = 0.95,
    clip_param: "(float) PPO parameter for clipping prob ratio" = 0.2,
    vfcoef: "(float) value function coefficient" = 0.5,
    entcoef: "(float) entropy coefficient" = 0.01,
    nminibatch: "(int) number of minibatches to break epoch of data into" = 4,
    n_epoch_vf: "(int) number of epochs to use when training the value function" = 1,
    n_epoch_pi: "(int) number of epochs to use when training the policy" = 1,
    lr: "(float) Adam learning rate" = 5e-4,
    default_loss_weights: "(dict) default_loss_weights" = {},
    store_segs: "(bool) whether or not to store segments in a buffer" = True,
    verbose: "(bool) print per-epoch loss stats" = True,
    log_save_opts: "(dict) passed into LogSaveHelper" = {},
    rnorm: "(bool) reward normalization" = True,
    kl_penalty: "(int) weight of the KL penalty, which can be used in place of clipping" = 0,
    grad_weight: "(float) relative weight of this worker's gradients" = 1,
    comm: "(MPI.Comm) MPI communicator" = None,
    callbacks: "(seq of function(dict)->bool) to run each update" = (),
    learn_state: "dict with optional keys {'opts', 'roller', 'lsh', 'reward_normalizer', 'curr_interact_count', 'seg_buf'}" = None,
):
    if comm is None:
        comm = MPI.COMM_WORLD

    learn_state = learn_state or {}
    ic_per_step = venv.num * comm.size * nstep

    opt_keys = (
        ["pi", "vf"] if (n_epoch_pi != n_epoch_vf) else ["pi"]
    )  # use separate optimizers when n_epoch_pi != n_epoch_vf
    params = list(model.parameters())
    opts = learn_state.get("opts") or {
        k: th.optim.Adam(params, lr=lr)
        for k in opt_keys
    }

    tu.sync_params(params)

    if rnorm:
        reward_normalizer = learn_state.get("reward_normalizer") or RewardNormalizer(venv.num)
    else:
        reward_normalizer = None

    def get_weight(k):
        return default_loss_weights[k] if k in default_loss_weights else 1.0

    def train_with_losses_and_opt(loss_keys, opt, **arrays):
        losses, diags = compute_losses(
            model,
            entcoef=entcoef,
            kl_penalty=kl_penalty,
            clip_param=clip_param,
            vfcoef=vfcoef,
            **arrays,
        )
        loss = sum([losses[k] * get_weight(k) for k in loss_keys])
        opt.zero_grad()
        loss.backward()
        tu.warn_no_gradient(model, "PPO")
        tu.sync_grads(params, grad_weight=grad_weight)
        diags = {k: v.detach() for (k, v) in diags.items()}
        opt.step()
        diags.update({f"loss_{k}": v.detach() for (k, v) in losses.items()})
        return diags

    def train_pi(**arrays):
        return train_with_losses_and_opt(["pi"], opts["pi"], **arrays)

    def train_vf(**arrays):
        return train_with_losses_and_opt(["vf"], opts["vf"], **arrays)

    def train_pi_and_vf(**arrays):
        return train_with_losses_and_opt(["pi", "vf"], opts["pi"], **arrays)

    roller = learn_state.get("roller") or Roller(
        act_fn=model.act,
        venv=venv,
        initial_state=model.initial_state(venv.num),
        keep_buf=100,
        keep_non_rolling=log_save_opts.get("log_new_eps", False),
    )

    lsh = learn_state.get("lsh") or LogSaveHelper(
        ic_per_step=ic_per_step, model=model, comm=comm, **log_save_opts
    )

    callback_exit = False  # Does callback say to exit loop?

    curr_interact_count = learn_state.get("curr_interact_count") or 0
    curr_iteration = 0
    seg_buf = learn_state.get("seg_buf") or []

    while curr_interact_count < interacts_total and not callback_exit:
        seg = roller.multi_step(nstep)
        lsh.gather_roller_stats(roller)
        if rnorm:
            seg["reward"] = reward_normalizer(seg["reward"], seg["first"])
        compute_advantage(model, seg, γ, λ, comm=comm)

        if store_segs:
            seg_buf.append(tree_map(lambda x: x.cpu(), seg))

        with logger.profile_kv("optimization"):
            # when n_epoch_pi != n_epoch_vf, we perform separate policy and vf epochs with separate optimizers
            if n_epoch_pi != n_epoch_vf:
                minibatch_optimize(
                    train_vf,
                    {k: seg[k] for k in INPUT_KEYS},
                    nminibatch=nminibatch,
                    comm=comm,
                    nepoch=n_epoch_vf,
                    verbose=verbose,
                )

                train_fn = train_pi
            else:
                train_fn = train_pi_and_vf

            epoch_stats = minibatch_optimize(
                train_fn,
                {k: seg[k] for k in INPUT_KEYS},
                nminibatch=nminibatch,
                comm=comm,
                nepoch=n_epoch_pi,
                verbose=verbose,
            )
            for (k, v) in epoch_stats[-1].items():
                logger.logkv("Opt/" + k, v)

        lsh()

        curr_interact_count += ic_per_step
        curr_iteration += 1

        for callback in callbacks:
            callback_exit = callback_exit or bool(callback(locals()))

    return dict(
        opts=opts,
        roller=roller,
        lsh=lsh,
        reward_normalizer=reward_normalizer,
        curr_interact_count=curr_interact_count,
        seg_buf=seg_buf,
    )
