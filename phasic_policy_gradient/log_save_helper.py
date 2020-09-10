import os
import time
import resource

import numpy as np
import torch as th
from . import logger
from mpi4py import MPI


def rcm(start, stop, modulus, mode="[)"):
    """
    Interval contains multiple, where 'mode' specifies whether it's
    closed or open on either side
    This was very tricky to get right
    """

    left_hit = start % modulus == 0
    middle_hit = modulus * (start // modulus + 1) < stop
    # ^^^ modulus * (start // modulus + 1) is the smallest multiple of modulus that's
    # strictly greater than start
    right_hit = stop % modulus == 0

    return (start < stop) and (
        (left_hit and mode[0] == "[") or (middle_hit) or (right_hit and mode[1] == "]")
    )

class LogSaveHelper:
    def __init__(
        self,
        model: "(nn.Module)",
        ic_per_step: "(int) number of iteractions per logging step",
        comm: "(MPI.Comm)" = None,
        ic_per_save: "(int) save only after this many interactions" = 100_000,
        save_mode: "(str) last: keep last model, all: keep all}" = "none",
        t0: "(float) override training start timestamp" = None,
        log_callbacks: "(list) extra callbacks to run before self.log()" = None,
        log_new_eps: "(bool) whether to log statistics for new episodes from non-rolling buffer" = False,
    ):
        self.model = model
        self.comm = comm or MPI.COMM_WORLD
        self.ic_per_step = ic_per_step
        self.ic_per_save = ic_per_save
        self.save_mode = save_mode
        self.save_idx = 0
        self.last_ic = 0
        self.log_idx = 0
        self.start_time = self.last_time = time.time()
        self.total_interact_count = 0
        if ic_per_save > 0:
            self.save()
        self.start_time = self.last_time = t0 or time.time()
        self.log_callbacks = log_callbacks
        self.log_new_eps = log_new_eps
        self.roller_stats = {}

    def __call__(self):
        self.total_interact_count += self.ic_per_step
        assert self.total_interact_count > 0, "Should start counting at 1"
        will_save = (self.ic_per_save > 0) and rcm(
            self.last_ic + 1, self.total_interact_count + 1, self.ic_per_save
        )
        self.log()
        if will_save:
            self.save()
        return True

    def gather_roller_stats(self, roller):
        self.roller_stats = {
            "EpRewMean": self._nanmean([] if roller is None else roller.recent_eprets),
            "EpLenMean": self._nanmean([] if roller is None else roller.recent_eplens),
        }
        if roller is not None and self.log_new_eps:
            assert roller.has_non_rolling_eps, "roller needs keep_non_rolling"
            ret_n, ret_mean, ret_std = self._nanmoments(roller.non_rolling_eprets)
            _len_n, len_mean, len_std = self._nanmoments(roller.non_rolling_eplens)
            roller.clear_non_rolling_episode_buf()
            self.roller_stats.update(
                {
                    "NewEpNum": ret_n,
                    "NewEpRewMean": ret_mean,
                    "NewEpRewStd": ret_std,
                    "NewEpLenMean": len_mean,
                    "NewEpLenStd": len_std,
                }
            )

    def log(self):
        if self.log_callbacks is not None:
            for callback in self.log_callbacks:
                callback()

        for k, v in self.roller_stats.items():
            logger.logkv(k, v)

        logger.logkv("Misc/InteractCount", self.total_interact_count)
        cur_time = time.time()
        Δtime = cur_time - self.last_time
        Δic = self.total_interact_count - self.last_ic

        logger.logkv("Misc/TimeElapsed", cur_time - self.start_time)
        logger.logkv("IPS_total", Δic / Δtime)
        logger.logkv("del_time", Δtime)
        logger.logkv("Iter", self.log_idx)
        logger.logkv(
            "CpuMaxMemory", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000
        )
        if th.cuda.is_available():
            logger.logkv("GpuMaxMemory", th.cuda.max_memory_allocated())
            th.cuda.reset_max_memory_allocated()

        if self.comm.rank == 0:
            print("RCALL_LOGDIR: ", os.environ["RCALL_LOGDIR"])
        logger.dumpkvs()
        self.last_time = cur_time
        self.last_ic = self.total_interact_count
        self.log_idx += 1

    def save(self):
        if self.comm.rank != 0:
            return
        if self.save_mode == "last":
            basename = "model"
        elif self.save_mode == "all":
            basename = f"model{self.save_idx:03d}"
        elif self.save_mode == "none":
            return
        else:
            raise NotImplementedError
        suffix = f"_rank{MPI.COMM_WORLD.rank:03d}" if MPI.COMM_WORLD.rank != 0 else ""
        basename += f"{suffix}.jd"
        fname = os.path.join(logger.get_dir(), basename)
        logger.log("Saving to ", fname, f"IC={self.total_interact_count}")
        th.save(self.model, fname, pickle_protocol=-1)
        self.save_idx += 1

    def _nanmean(self, xs):
        xs = _flatten(self.comm.allgather(xs))
        return np.nan if len(xs) == 0 else np.mean(xs)

    def _nanmoments(self, xs, **kwargs):
        xs = _flatten(self.comm.allgather(xs))
        return _nanmoments_local(xs, **kwargs)


def _flatten(ls):
    return [el for sublist in ls for el in sublist]


def _nanmoments_local(xs, ddof=1):
    n = len(xs)
    if n == 0:
        return n, np.nan, np.nan
    elif n == ddof:
        return n, np.mean(xs), np.nan
    else:
        return n, np.mean(xs), np.std(xs, ddof=ddof)
