from os import walk, sep
from numpy import loadtxt


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
            for fname in fnames:
                temp = dict()  # for temporarily storing data
                if 'annotations' in fname:
                    if segment:
                        events, starts, stops = loadtxt(dname + sep + fname, dtyp=float, delimiter=',', skiprows=1,
                                                        usecols=(2, 4, 5), unpack=True)
                else:
                    temp[fname[:-3]] = loadtxt(dname + sep + fname, dtype=float, delimiter=',', skiprows=1)


    return walk_dir
