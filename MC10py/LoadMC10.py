from os import walk, sep, listdir
from numpy import loadtxt, ndarray, repeat, argmin, unique, full_like, zeros, argwhere, arange, mean, diff, interp,\
    array


# TODO add option to create separate files for each subject
# TODO add folder size check for memory issues (ie force separate files for each subject)
def load_mc10(study_dir, segment=True, sync=True):
    """
    Load raw MC10 data from a study directory containing folders for each subjected as downloaded from the
    BioStamp RC web portal

    Parameters
    ---------
    study_dir : str
        Base study directory containing subject folders.
    segment : bool, optional
        Segment the data based on annotations.  Defaults to True.
    sync : bool, optional
        Synchronize the timestamps for the inertial sensors.  Timestamps for sensors with the same sampling rates
        will be the same.  All sensors will start at the same time, regardless of sampling rate.

    Returns
    -------
    data : dict
        Loaded data.  Top down structure is 'subject_id', 'sensor location', 'event id' (if segmenting), 'sensor_type'
    """

    bn = len(study_dir.split(sep))  # base length of study folder

    subjs = list(listdir(study_dir))  # list all subfolders, which are subject IDs

    data = {i: dict() for i in subjs}  # allocate data storage for each subject

    for sub in subjs:
        print(f'{sub}\n ---------')
        wkd = walk(study_dir + sep + sub)  # walk down each subject folder
        temp = dict()
        for dname, _, fnames in wkd:
            if 'annotations.csv' in fnames and segment:
                # import annotations for data segmenting
                events, starts, stops = loadtxt(dname + sep + 'annotations.csv', dtype='U35',
                                                delimiter='","', skiprows=1, usecols=(2, 4, 5), unpack=True)
                # checking for non-unique event names
                uniq, inds, cnts = unique(events, return_counts=True, return_inverse=True)
                if any(cnts > 1):  # if any events have more than 1 count
                    ecnts = full_like(cnts, 1)  # create array to keep track of number to add
                    for k in range(len(inds)):  # iterate over the events
                        if cnts[inds[k]] > 1:  # if this event has more than one occurance
                            events[k] += f' {ecnts[inds[k]]}'  # add the chronological number occurence it is to name
                            ecnts[inds[k]] += 1  # increment the occurence number tracker
            elif 'accel.csv' in fnames:
                sens_loc = dname.split(sep)[bn+1]  # get the sensor location from the directory name
                temp[sens_loc] = dict()  # make a sub dictionary for the sensor location
                for fname in fnames:  # can be multiple files per location (ie accel and gyro)
                    if 'errors' not in fname:  # don't want to do anything with *_error.csv files
                        # load data into data dictionary
                        print(sens_loc, fname)
                        temp[sens_loc][fname[:-4]] = loadtxt(dname + sep + fname, dtype=float, delimiter=',',
                                                             skiprows=1)
        if sync:
            temp = _align_timestamps(temp)  # align time stamps of data
        if segment:
            data[sub] = _segment_data(temp, starts, stops, events)  # segment the data
        else:
            data[sub] = temp
    return data


class InputError(Exception):
    pass


def _segment_data(data, start, stop, events):
    """
    Segments raw data into specific segments.  These are typically reported in the annotations.csv file for each
    subject

    Parameters
    ----------
    data : dict
        Dictionary of MxN arrays of data for each sensor location, where M is the number of data points and N
        is the number of columns, typically time, and then 3 axes for inertial sensors, or time and voltage for EMG/ECG.
        Note that time MUST be the first column.
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

    for sens_loc in data.keys():
        split_data[sens_loc] = dict()
        for typ in data[sens_loc].keys():
            split_data[sens_loc][typ] = dict()
            # extract time data and repeat it for all events
            times = repeat(data[sens_loc][typ][:, 0].reshape((-1, 1)), start.size, axis=1)

            start_inds = argmin(abs(times-start), axis=0)  # get the indices for the start times
            stop_inds = argmin(abs(times-stop), axis=0)  # get the indices for the stop times

            for ib, ie, ev in zip(start_inds, stop_inds, events):
                split_data[sens_loc][typ][ev] = data[sens_loc][typ][ib:ie, :]  # segment out the data of interest

    return split_data


def _align_timestamps(subj_data):
    """
    Align timestamps across multiple sensors for the same subject.  Currently only works for inertial sensors

    subj_data : dict
        Dictionary of all data for subject
    """

    sampling_rates = 1000/array([31.25, 62.5, 125, 250])  # valid sampling rates in the MC10 system currently

    locs = array(list(subj_data.keys()))  # keep a list of the sensor locations

    tb = zeros((len(locs),))
    tf = zeros((len(locs),))

    for loc, i in zip(locs, range(len(locs))):
        if 'accel' in subj_data[loc].keys():
            tb[i] = subj_data[loc]['accel'][0, 0]  # get first timestamp
            tf[i] = subj_data[loc]['accel'][-1, 0]  # get last timestamp

    nz = argwhere(tb)  # locations where there is an inertial sensor
    locs = locs[nz].flatten()  # cut down locations to only those with accel readings
    tb = tb[nz]  # cut down locations to only those with accel readings
    tf = tf[nz]  # cut down locations to only those with accel readings

    start = max(tb)  # get the maximum begin time and use it for the start
    stop = min(tf)  # get the minimum end time and use if for the stop

    ret_data = dict()  # data to be returned
    for loc in locs:
        dt = sampling_rates[argmin(abs(mean(diff(subj_data[loc]['accel'][:, 0])) - sampling_rates))]
        time = arange(start, stop, dt)  # create timestamps.
        # Created per location in case of different sampling rates

        ret_data[loc] = dict()
        for sens in subj_data[loc].keys():
            ret_data[loc][sens] = zeros((len(time), len(subj_data[loc][sens][0, :])))
            for i in range(1, len(subj_data[loc][sens][0, :])):
                ret_data[loc][sens][:, i] = interp(time, subj_data[loc][sens][:, 0], subj_data[loc][sens][:, i])
            ret_data[loc][sens][:, 0] = time

    return ret_data





