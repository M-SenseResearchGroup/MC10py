from os import walk, sep
from numpy import loadtxt, ndarray, repeat, argmin, unique, full_like


# TODO add option to create separate files for each subject
# TODO add folder size check for memory issues (ie force separate files for each subject)
def load_mc10(study_dir, segment=True):
    """
    Load raw MC10 data from a study directory containing folders for each subjected as downloaded from the
    BioStamp RC web portal

    Parameters
    ---------
    study_dir : str
        Base study directory containing subject folders.
    segment : bool, optional
        Segment the data based on annotations.  Defaults to True.

    Returns
    -------
    """

    bn = len(study_dir.split(sep))  # base length of study folder

    walk_dir = list(walk(study_dir))  # walk along the provided path to get all the files needed.
    # returns a generator that yields (dirpath, dirnames, filenames)
    subjs = walk_dir[0][1]  # in the base folder, in the dirnames result

    data = {i: dict() for i in subjs}  # allocate data storage for each subject

    for dname, _, fnames in walk_dir:
        if fnames != []:
            subj = dname.split(sep)[bn]
            temp = dict()  # for temporarily storing data
            print(subj, fnames)
            if 'annotations.csv' in fnames and segment:
                events, starts, stops = loadtxt(dname + sep + 'annotations.csv', dtype='U25',
                                                delimiter='","', skiprows=1, usecols=(2, 4, 5), unpack=True)
                uniq, inds, cnts = unique(events, return_counts=True, return_inverse=True)
                if any(cnts > 1):
                    ecnts = full_like(cnts, 1)
                    for k in range(len(inds)):
                        if cnts[inds[k]] > 1:
                            events[k] += f' {ecnts[inds[k]]}'
                            ecnts[inds[k]] += 1
            else:
                sens_loc = dname.split(sep)[bn + 1]
                temp[sens_loc] = dict()
                data[subj][sens_loc] = dict()
                for fname in fnames:
                    if 'errors' not in fname:
                        temp[sens_loc][fname[:-4]] = loadtxt(dname + sep + fname, dtype=float, delimiter=',',
                                                             skiprows=1)
                        data[subj][sens_loc][fname[:-4]] = _segment_data(temp[sens_loc][fname[:-4]], starts, stops,
                                                                         events)

    return data


class InputError(Exception):
    pass


def _segment_data(data, start, stop, events):
    """
    Segments raw data into specific segments.  These are typically reported in the annotations.csv file for each
    subject

    Parameters
    ----------
    data : array_like
        MxN array of data, where M is the number of data points and N is the number of columns, typically time, and
        then 3 axes for inertial sensors, or time and voltage for EMG/ECG.  Note that time MUST be the first column.
    start : float, int, array_like
        Single start timestamp, or 1D array of start timestamps. 'start' and 'stop' must be the same length
    stop : float, int, array_like
        Single stop timestamp, or 1D array of stop timestamps.  'start' and 'stop' must be the same length
    events : str, array_like
        Single event name, or 1D array of event names.  Must be the same length as 'start'

    Returns
    -------
    split_data : dict
        Dict of different events and their corresponding data, with keys being provided from 'id'
    """

    # test input required conditions
    if not isinstance(start, (float, int, list, ndarray)):
        raise InputError("""'start' must be a float, int, list, or Numpy array""")
    if not isinstance(stop, (float, int, list, ndarray)):
        raise InputError("""'stop' must be a float, int, list, or Numpy array""")

    start = start.astype(float, copy=False)
    stop = stop.astype(float, copy=False)

    if start.ndim != 1:
        raise InputError("""'start' must be a 1-D array.""")
    if stop.ndim != 1:
        raise InputError("""'stop must be a 1-D array.""")
    if start.shape != stop.shape:
        raise InputError("""'start' and 'stop' must be the same shape.""")
    if start.shape != events.shape:
        raise InputError("""'events' must be the same length as 'start' and 'stop'.""")

    split_data = dict()

    times = repeat(data[:, 0].reshape((-1, 1)), start.size, axis=1)  # extract time data and repeat it for all events

    start_inds = argmin(abs(times-start), axis=0)  # get the indices for the start times
    stop_inds = argmin(abs(times-stop), axis=0)  # get the indices for the stop times

    for ib, ie, ev in zip(start_inds, stop_inds, events):
        split_data[ev] = data[ib:ie, :]  # segment out the data of interest

    return split_data

