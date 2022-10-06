def get_disklab_main_directory():
    """
    If you install DISKLAB and you do not know where the code and the
    packages are put, you can call:

       import disklab
       dir = disklab.utilities.get_disklab_main_directory()

    The dir is then a string which is the path to the DISKLAB stuff.
    """
    import pkg_resources
    import os
    return os.path.realpath(pkg_resources.resource_filename(__name__, os.pardir))


def hash_arrays(a):
    """
    Return a unique hash of a numpy array or a set of numpy arrays.
    This can be useful for testing.
    """
    import hashlib
    if type(a) == tuple:
        content = b''
        for b in a:
            content += b.tobytes()
        hashhex = hashlib.sha1(content).hexdigest()
    else:
        content = a.tobytes()
        hashhex = hashlib.sha1(content).hexdigest()
    return hashhex
