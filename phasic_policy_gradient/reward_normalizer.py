import torch as th
import torch.distributed as dist
from torch import nn

from . import torch_util as tu


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(
        self,
        epsilon: "initial count (with mean=0 ,var=1)" = 1e-4,
        shape: "unbatched shape of data" = (),
        distributed: "whether to allreduce stats" = True,
    ):
        super().__init__()
        self.register_buffer("mean", th.zeros(shape))
        self.register_buffer("var", th.ones(shape))
        self.register_buffer("count", th.tensor(epsilon))
        self.distributed = distributed and tu.is_distributed()

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = th.tensor([x.shape[0]], device=x.device, dtype=th.float32)
        if self.distributed:
            # flatten+unflatten so we just need one allreduce
            flat = tu.flatten_tensors([batch_mean, batch_var, batch_count])
            flat = flat.to(device=tu.dev())  # Otherwise all_mean_ will fail
            tu.all_mean_(flat)
            tu.unflatten_to(flat, [batch_mean, batch_var, batch_count])
            batch_count *= dist.get_world_size()
        self.update_from_moments(batch_mean, batch_var, batch_count[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # pylint: disable=attribute-defined-outside-init
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RewardNormalizer:
    """
    Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
    section 9.3 (which is based on our Baselines code, haha)
    Motivation is that we'd rather normalize the returns = sum of future rewards,
    but we haven't seen the future yet. So we assume that the time-reversed rewards
    have similar statistics to the rewards, and normalize the time-reversed rewards.
    """

    def __init__(self, num_envs, cliprew=10.0, gamma=0.99, epsilon=1e-8, per_env=False):
        ret_rms_shape = (num_envs,) if per_env else ()
        self.ret_rms = RunningMeanStd(shape=ret_rms_shape)
        self.cliprew = cliprew
        self.ret = th.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        rets = backward_discounted_sum(
            prevret=self.ret, reward=reward.cpu(), first=first.cpu(), gamma=self.gamma
        )
        self.ret = rets[:, -1]
        self.ret_rms.update(rets if self.per_env else rets.reshape(-1))
        return self.transform(reward)

    def transform(self, reward):
        return th.clamp(
            reward / th.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


def backward_discounted_sum(
    *,
    prevret: "(th.Tensor[1, float]) value predictions",
    reward: "(th.Tensor[1, float]) reward",
    first: "(th.Tensor[1, bool]) mark beginning of episodes",
    gamma: "(float)",
):
    first = first.to(dtype=th.float32)
    assert first.dim() == 2
    _nenv, nstep = reward.shape
    ret = th.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[:, t] = reward[:, t] + (1 - first[:, t]) * gamma * prevret
    return ret
