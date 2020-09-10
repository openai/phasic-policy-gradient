import warnings
from functools import partial
import torch as th
import torch.distributions as dis
from gym3.types import Discrete, Real, TensorType

def _make_categorical(x, ncat, shape):
    x = x.reshape((*x.shape[:-1], *shape, ncat))
    return dis.Categorical(logits=x)


def _make_normal(x, shape):
    warnings.warn("Using stdev=1")
    return dis.Normal(loc=x.reshape(x.shape[:-1] + shape), scale=1.0)


def _make_bernoulli(x, shape):  # pylint: disable=unused-argument
    return dis.Bernoulli(logits=x)


def tensor_distr_builder(ac_space):
    """
    Like distr_builder, but where ac_space is a TensorType
    """
    assert isinstance(ac_space, TensorType)
    eltype = ac_space.eltype
    if eltype == Discrete(2):
        return (ac_space.size, partial(_make_bernoulli, shape=ac_space.shape))
    if isinstance(eltype, Discrete):
        return (
            eltype.n * ac_space.size,
            partial(_make_categorical, shape=ac_space.shape, ncat=eltype.n),
        )
    else:
        raise ValueError(f"Expected ScalarType, got {type(ac_space)}")


def distr_builder(ac_type) -> "(int) size, (function) distr_from_flat":
    """
    Tell a network constructor what it needs to produce a certain output distribution
    Returns:
        - size: the size of a flat vector needed to construct the distribution
        - distr_from_flat: function that takes flat vector and turns it into a
          torch.Distribution object.
    """
    if isinstance(ac_type, TensorType):
        return tensor_distr_builder(ac_type)
    else:
        raise NotImplementedError
