import torch as th
from . import logger
from .tree_util import tree_map
from . import torch_util as tu

def _fmt_row(width, row, header=False):
    out = " | ".join(_fmt_item(x, width) for x in row)
    if header:
        out = out + "\n" + "-" * len(out)
    return out


def _fmt_item(x, l):
    if th.is_tensor(x):
        assert x.dim() == 0
        x = float(x)
    if isinstance(x, float):
        v = abs(x)
        if (v < 1e-4 or v > 1e4) and v > 0:
            rep = "%7.2e" % x
        else:
            rep = "%7.5f" % x
    else:
        rep = str(x)
    return " " * (l - len(rep)) + rep


class LossDictPrinter:
    """
    Helps with incrementally printing out stats row by row in a formatted table
    """

    def __init__(self):
        self.printed_header = False

    def print_row(self, d):
        if not self.printed_header:
            logger.log(_fmt_row(12, d.keys()))
            self.printed_header = True
        logger.log(_fmt_row(12, d.values()))

def minibatch_optimize(
    train_fn: "function (dict) -> dict called on each minibatch that returns training stats",
    tensordict: "Dict[str->th.Tensor]",
    *,
    nepoch: "(int) number of epochs over dataset",
    nminibatch: "(int) number of minibatch per epoch",
    comm: "(MPI.Comm) MPI communicator",
    verbose: "(bool) print detailed stats" = False,
    epoch_fn: "function () -> dict to be called each epoch" = None,
):
    ntrain = tu.batch_len(tensordict)
    if nminibatch > ntrain:
        logger.log(f"Warning: nminibatch > ntrain!! ({nminibatch} > {ntrain})")
        nminibatch = ntrain
    ldp = LossDictPrinter()
    epoch_dicts = []
    for _ in range(nepoch):
        mb_dicts = [
            train_fn(**mb) for mb in minibatch_gen(tensordict, nminibatch=nminibatch)
        ]
        local_dict = {k: float(v) for (k, v) in dict_mean(mb_dicts).items()}
        if epoch_fn is not None:
            local_dict.update(dict_mean(epoch_fn()))
        global_dict = dict_mean(comm.allgather(local_dict))
        epoch_dicts.append(global_dict)
        if verbose:
            ldp.print_row(global_dict)
    return epoch_dicts


def dict_mean(ds):
    return {k: sum(d[k] for d in ds) / len(ds) for k in ds[0].keys()}


def to_th_device(x):
    assert th.is_tensor(x), "to_th_device should only be applied to torch tensors"
    dtype = th.float32 if x.dtype == th.float64 else None
    return x.to(tu.dev(), dtype=dtype)


def minibatch_gen(data, *, batch_size=None, nminibatch=None, forever=False):
    assert (batch_size is None) != (
        nminibatch is None
    ), "only one of batch_size or nminibatch should be specified"
    ntrain = tu.batch_len(data)
    if nminibatch is None:
        nminibatch = max(ntrain // batch_size, 1)
    while True:
        for mbinds in th.chunk(th.randperm(ntrain), nminibatch):
            yield tree_map(to_th_device, tu.tree_slice(data, mbinds))
        if not forever:
            return
