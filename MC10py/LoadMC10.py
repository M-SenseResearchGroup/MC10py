from os import walk, sep
from numpy import loadtxt, array


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
            if 'annotations.csv' in fnames and segment:
                events, starts, stops = loadtxt(dname + sep + 'annotations.csv', dtype=str, delimiter=',', skiprows=1,
                                                usecols=(2, 4, 5), unpack=True)
            else:
                sens_loc = dname.split(sep)[bn + 1]
                temp[sens_loc] = dict()
                for fname in fnames:
                    if 'errors' not in fname:
                        temp[sens_loc][fname[:-4]] = loadtxt(dname + sep + fname, dtype=float, delimiter=',',
                                                             skiprows=1)

    return temp


class InputError(Exception):
    pass


def _segment_data(data, start, stop, id):
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
    id : str, array_like
        Single event name, or 1D array of event names.  Must be the same length as 'start'

    Returns
    -------
    split_data : dict
        Dict of different events and their corresponding data, with keys being provided from 'id'
    """

    # test input required conditions
    if not isinstance(start, (float, int, list, array)):
        raise InputError("""'start' must be a float, int, list, or Numpy array""")
    if not isinstance(stop, (float, int, list, array)):
        raise InputError("""'stop' must be a float, int, list, or Numpy array""")
