import numpy as np
import pandas as pd
import re
import h5py
import datetime
from pathlib import Path

# from Labber import LogFile

sci_dict = {"u": 1e-6, "m": 1e-3, "": 1, "k": 1e3, "M": 1e6, "G": 1e9}


def load_h5_file(f_name):
    """
    Return an h5py file instance from the f_name.
    """
    if not isinstance(f_name, h5py.File):
        f_name = Path(f_name)
        f = h5py.File(f_name)
    else:
        f = f_name
    return f


def print_h5_structure(h5_file, indent: int = 0):
    """
    Recursively lists keys and attributes of an HDF5 file with indentation.

    :param hdf5_file: HDF5 file object or group within the file.
    :param indent: Current level of indentation (used for recursive calls).
    """
    # Indentation string
    indent_str = "\t" * indent

    if indent == 0:
        for attr, value in h5_file.attrs.items():
            if attr == "comment":
                print(f"File Attribute: '{attr}', Value: \n{value}")
            else:
                print(f"File Attribute: '{attr}', Value: {value}")

    # Iterate through keys in the file or group
    for key in h5_file.keys():
        print(f"{indent_str} - Key: {key}")

        # Print attributes for each key
        for attr, value in h5_file[key].attrs.items():
            print(f"{indent_str} - Attribute: '{attr}', Value: {value}")

        # If the current key is a dataset, list its shape
        if isinstance(h5_file[key], h5py.Dataset):
            print(f"{indent_str} - Type: Dataset, Shape: {h5_file[key].shape}")
            # Also print its datatype as an array line by line
            dtype_str = str(h5_file[key].dtype).replace("), (", f")\n {indent_str} (")
            print(f"{indent_str} - Datatype: \n {indent_str} {dtype_str}")

        # If the current key is a group, recursively list its contents
        if isinstance(h5_file[key], h5py.Group):
            print_h5_structure(h5_file[key], indent + 1)


def nearest(items, pivot):
    return items.index(min(items, key=lambda x: abs(x - pivot)))


def resample(trace, points):
    # Resample a trace into x points,
    # the original samples in the xth new point are summarised by a mean and std
    assert isinstance(points, int)
    assert len(trace) > points
    div = int(np.floor(len(trace) / points))
    matrix = np.reshape(trace[: div * points], (points, div))
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    return means, stds


def extract_sci(raw_str):
    multiplier = re.findall("[umkMG]", raw_str)
    multiplier.append("")
    sci_not = float(re.findall("[0-9]+", raw_str)[0]) * sci_dict[multiplier[0]]
    return sci_not


### Wrapper functions for Labber API ###


def getChannelValuesAsDict(h5_file):
    """
    Get the saved instrument configuration for each channel as a dictionary.
    """
    h5_file = load_h5_file(h5_file)
    channels = np.array(h5_file["Channels"])
    channel_values = h5_file["Instrument config"]
    channel_info = {}

    for ch in channels:
        channel_val = channel_values.get(ch[1]).attrs.get(ch[2])
        channel_info[ch[0].decode()] = channel_val
    return channel_info


def getTime(h5_file):
    # Return a datetime object without the trailing microseconds
    h5_file = load_h5_file(h5_file)
    timestamp = h5_file.attrs["creation_time"]
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    dt_object -= datetime.timedelta(microseconds=dt_object.microsecond)
    return dt_object


def getLogChannels(h5_file):
    """
    Return the channels logged as output.
    """
    h5_file = load_h5_file(h5_file)
    log_channels = []
    for n in h5_file["Log list"][:]["channel_name"]:
        d = {"name": n}
        # Use the first found item (should only have one)
        mask = h5_file["Channels"][:]["name"] == n
        d["unit"] = h5_file["Channels"][:]["unitInstr"][mask][0].decode()
        # Check if this channel is in trace:
        if n in h5_file["Traces"].keys():
            d["vector"] = True
            d["complex"] = h5_file["Traces"][n].attrs["complex"]
        else:
            d["vector"] = False
            d["complex"] = False
        log_channels.append(d)
    return log_channels


def getStepChannels(h5_file):
    """
    Get the step information.
    """
    h5_file = load_h5_file(h5_file)
    step_dims = h5_file.attrs["Step dimensions"]
    steps = np.array(h5_file["Step list"])
    step_values = h5_file["Step config"]
    step_info = []

    for st, dims in zip(steps, step_dims):
        # For now, do not support step relations.
        if st[5]:
            print(st[0].decode(), " -> Step relations not supported, skipping.")
            continue

        if dims > 1:
            d = {"name": st[0].decode()}
            mask = h5_file["Channels"][:]["name"] == st[0]
            d["unit"] = h5_file["Channels"][:]["unitInstr"][mask][0].decode()
            
            # Labber will save the correct number of steps
            step_val_start = np.array(
                step_values.get(st[0])["Step items"].fields("start")
            )
            step_val_stop = np.array(
                step_values.get(st[0])["Step items"].fields("stop")
            )
            step_val_npts = np.array(
                step_values.get(st[0])["Step items"].fields("n_pts")
            )

            step_val = []
            last_stop = np.inf
            for start, stop, npts in zip(
                step_val_start,
                step_val_stop,
                step_val_npts,
            ):
                step_val_tmp = np.linspace(start, stop, num=npts)
                # check if the start is the same as the last stop
                # If so, remove the first element
                if start == last_stop:
                    step_val_tmp = step_val_tmp[1:]
                step_val.append(step_val_tmp)
                last_stop = stop
            d["values"] = np.hstack(step_val)
            step_info.append(d)

    # print(step_info)
    return step_info


def getStepNames(h5_file):
    """
    Get the channel names of stepped channels.
    """
    h5_file = load_h5_file(h5_file)
    return list(h5_file["Data"]["Channel names"].fields("name"))


def getXdim(h5_file, log_ch, tr_nr=0):
    h5_file = load_h5_file(h5_file)
    if type(log_ch) != bytes:
        log_ch = str.encode(log_ch)
    npts = h5_file["Traces"][log_ch + b"_N"]
    if npts.len() == 1:
        return npts[0]
    else:
        return npts[tr_nr]


def getXaxis(h5_file, log_ch, tr_nr=0):
    h5_file = load_h5_file(h5_file)
    if type(log_ch) != bytes:
        log_ch = str.encode(log_ch)
    if h5_file["Traces"][log_ch + b"_t0dt"].len() == 1:
        t0, dt = h5_file["Traces"][log_ch + b"_t0dt"][0]
    else:
        t0, dt = h5_file["Traces"][log_ch + b"_t0dt"][tr_nr]
    if h5_file["Traces"][log_ch + b"_N"].len() == 1:
        npts = h5_file["Traces"][log_ch + b"_N"][0]
    else:
        npts = h5_file["Traces"][log_ch + b"_N"][tr_nr]
    return np.arange(npts) * dt + t0


def getNumberOfEntries(h5_file):
    h5_file = load_h5_file(h5_file)
    return len(getTimestamps(h5_file))


def getTimestamps(h5_file):
    h5_file = load_h5_file(h5_file)
    return np.array(h5_file["Traces"]["Time stamp"])


def getTrace(h5_file, log_ch, tr_nr=0):
    h5_file = load_h5_file(h5_file)
    if not isinstance(log_ch, bytes):
        log_ch = str.encode(log_ch)
    is_complex = h5_file["Traces"][log_ch].attrs["complex"]
    trace = np.empty(getXdim(h5_file, log_ch, tr_nr))
    h5_file["Traces"][log_ch].read_direct(trace, source_sel=np.s_[:, 0, tr_nr])
    if is_complex:
        trace_c = np.empty(getXdim(h5_file, log_ch, tr_nr))
        h5_file["Traces"][log_ch].read_direct(trace_c, source_sel=np.s_[:, 1, tr_nr])
        trace = trace.astype(complex) + trace_c * 1j
    return trace


def getData(h5_file, log_ch):
    h5_file = load_h5_file(h5_file)
    if not isinstance(log_ch, bytes):
        log_ch = str.encode(log_ch)
    is_complex = h5_file["Traces"][log_ch].attrs["complex"]
    trace = np.empty((getXdim(h5_file, log_ch), getNumberOfEntries(h5_file)))
    h5_file["Traces"][log_ch].read_direct(trace, source_sel=np.s_[:, 0, :])
    if is_complex:
        print("Complex data found.")
        trace_c = np.empty((getXdim(h5_file, log_ch), getNumberOfEntries(h5_file)))
        h5_file["Traces"][log_ch].read_direct(trace_c, source_sel=np.s_[:, 1, :])
        trace = trace.astype(complex) + trace_c * 1j
    return trace.T


def getSteps(h5_file):
    h5_file = load_h5_file(h5_file)
    steps = np.swapaxes(h5_file["Data"]["Data"], 1, 2)
    return np.reshape(steps, (-1, len(getStepNames(h5_file))), order="F")


def getComment(h5_file):
    h5_file = load_h5_file(h5_file)
    return h5_file.attrs["comment"]


# def getXYTraceSteps(h5_file, log_ch, tr_nr=0, is_complex=False):
#     h5_file = load_h5_file(h5_file)
#     return (
#         getXaxis(h5_file, log_ch, tr_nr),
#         getTrace(h5_file, log_ch, tr_nr, is_complex),
#         getSteps(h5_file)[tr_nr],
#     )


# col1 is the x-axis, col2 the y-axis
def prepare_2D(h5_file, col1, col2, data, return_max=False):
    debug_str = ""
    steps = getSteps(h5_file)
    stepnames = getStepNames(h5_file)
    steps_df = pd.DataFrame(steps, columns=stepnames)
    x = steps_df[col1].unique()
    y = steps_df[col2].unique()
    # col1_idx = steps_df.columns.get_loc(col1)
    # col2_idx = steps_df.columns.get_loc(col2)
    if len(x) * len(y) == len(data):
        debug_str += "No accumulation needed."
    else:
        debug_str += "Ambiguity, take mean over grouped values. "
    steps_df["data"] = data
    data_2d = (
        steps_df[[col1, col2, "data"]].groupby([col2, col1]).mean().unstack().values
    )
    print(debug_str)
    if return_max:
        return x, y, data_2d, steps_df.iloc[steps_df["data"].idxmax()]
    else:
        return x, y, data_2d
