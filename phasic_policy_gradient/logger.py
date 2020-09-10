import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, wraps
from mpi4py import MPI


def mpi_weighted_mean(comm, local_name2valcount):
    """
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    local_name2valcount = {
        name: (float(val), count)
        for (name, (val, count)) in local_name2valcount.items()
    }
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                name2sum[name] += val * count
                name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


class KVWriter(ABC):
    @abstractmethod
    def writekvs(self, kvs):
        pass

    @abstractmethod
    def close(self):
        pass


class SeqWriter(ABC):
    @abstractmethod
    def writeseq(self, seq):
        pass

    @abstractmethod
    def close(self):
        pass


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = []
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str.append((self._truncate(key), self._truncate(valstr)))

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(lambda kv: len(kv[0]), key2str))
            valwidth = max(map(lambda kv: len(kv[1]), key2str))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str, key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if hasattr(v, "__float__"):
                v = float(v)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)

        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def logkvs_mean(d):
    """
    Log a dictionary of key-value pairs with averaging over multiple calls
    """
    for (k, v) in d.items():
        logkv_mean(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


def log(*args):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args)


def warn(*args):
    get_current().warn(*args)


def get_dir():
    """
    Get directory that log files are being written to.
    Will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


@contextmanager
def profile_kv(scopename, sync_cuda=False):
    if sync_cuda:
        _sync_cuda()
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        if sync_cuda:
            _sync_cuda()
        get_current().name2val[logkey] += time.time() - tstart


def _sync_cuda():
    from torch import cuda

    cuda.synchronize()


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func, name):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            with profile_kv(name):
                return func(*args, **kwargs)

        return func_wrapper

    if callable(n):
        return decorator_with_name(n, n.__name__)
    elif isinstance(n, str):
        return partial(decorator_with_name, name=n)
    else:
        raise NotImplementedError(
            "profile should be called as either a bare decorator"
            " or with a string (profiling name of a function) as an argument"
        )


def dump_kwargs(func):
    """
    Prints all keyword-only parameters of a function. Useful to print hyperparameters used.
    Usage:
    @logger.dump_kwargs
    def create_policy(*, hp1, hp2, hp3): ...
    or
    logger.dump_kwargs(ppo.learn)(lr=60e-5, ...)
    """

    def func_wrapper(*args, **kwargs):
        import inspect, textwrap

        sign = inspect.signature(func)
        for k, p in sign.parameters.items():
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                default = "%15s (default)" % str(sign.parameters[k].default)
                get_current().log(
                    "%s.%s: %15s = %s"
                    % (
                        func.__module__,
                        func.__qualname__,
                        k,
                        textwrap.shorten(
                            str(kwargs.get(k, default)),
                            width=70,
                            drop_whitespace=False,
                            placeholder="...",
                        ),
                    )
                )
        return func(*args, **kwargs)

    return func_wrapper


# ================================================================
# Backend
# ================================================================

# Pytorch explainer:
# If you keep a reference to a variable that depends on parameters, you
# keep around the whole computation graph. That causes an unpleasant surprise
# if you were just trying to log a scalar. We could cast to float, but
# that would require a synchronization, and it would be nice if logging
# didn't require the value to be available immediately. Therefore we
# detach the value at the point of logging, and only cast to float when
# dumping to the log file.


def get_current():
    if not is_configured():
        raise Exception("you must call logger.configure() before using logger")
    return Logger.CURRENT


class Logger(object):
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        if hasattr(val, "requires_grad"):  # see "pytorch explainer" above
            val = val.detach()
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        assert hasattr(val, "__float__")
        if hasattr(val, "requires_grad"):  # see "pytorch explainer" above
            val = val.detach()
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if self.comm.rank == 0:
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args):
        self._do_log(args)

    def warn(self, *args):
        self._do_log(("[WARNING]", *args))

    # Configuration
    # ----------------------------------------
    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def configure(
    dir: "(str|None) Local directory to write to" = None,
    format_strs: "(str|None) list of formats" = None,
    comm: "(MPI communicator | None) average numerical stats over comm" = None,
):
    if dir is None:
        if os.getenv("OPENAI_LOGDIR"):
            dir = os.environ["OPENAI_LOGDIR"]
        else:
            dir = osp.join(
                tempfile.gettempdir(),
                datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
            )
    os.makedirs(dir, exist_ok=True)

    # choose log suffix based on world rank because otherwise the files will collide
    # if we split the world comm into different comms
    if MPI.COMM_WORLD.rank == 0:
        log_suffix = ""
    else:
        log_suffix = "-rank%03i" % MPI.COMM_WORLD.rank

    if comm is None:
        comm = MPI.COMM_WORLD

    format_strs = format_strs or default_format_strs(comm.rank)

    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    log("logger: logging to %s" % dir)


def is_configured():
    return Logger.CURRENT is not None


def default_format_strs(rank):
    if rank == 0:
        return ["stdout", "log", "csv"]
    else:
        return []


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger


# ================================================================


def _demo():
    configure()
    log("hi")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    log("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    log("^^^ should see b = -33.3")

    logkv("b", -2.5)
    dumpkvs()


# ================================================================
# Readers
# ================================================================


def read_json(fname):
    import pandas

    ds = []
    with open(fname, "rt") as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)


def read_csv(fname):
    import pandas

    return pandas.read_csv(fname, index_col=None, comment="#")


def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    import tensorflow as tf

    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError(
            "Expected tensorboard file or directory containing them. Got %s" % path
        )
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx, tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step - 1, colidx] = value
    return pandas.DataFrame(data, columns=tags)


if __name__ == "__main__":
    _demo()
