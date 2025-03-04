import numpy as np
import starsim as ss

class MultiFloat(ss.Arr):
    def __init__(self, name=None, nan=np.nan, **kwargs):
        super().__init__(name=name, dtype=np.ndarray, nan=nan, **kwargs)
        return

    def set(self, uids, new_vals=None):
        """ Set the values for the specified UIDs"""
        if new_vals is None:
            if isinstance(self.default, ss.Dist):
                new_vals = self.default.rvs(uids)
            elif callable(self.default):
                new_vals = self.default(len(uids))
            elif self.default is not None:
                new_vals = self.default
            else:
                new_vals = self.nan
        for uid in uids:
            self.raw[uid] = new_vals.copy()
        return new_vals


    def asnew(self, arr=None, cls=None, name=None):
        """ Duplicate and copy (rather than link) data, optionally resetting the array """
        if cls is None:
            cls = self.__class__
        if arr is None:
            arr = self.values
        new = object.__new__(cls) # Create a new Arr instance
        new.__dict__ = self.__dict__.copy() # Copy pointers
        new.dtype = arr.dtype # Set to correct dtype
        new.name = name # In most cases, the asnew Arr has different values to the original Arr so the original name no longer makes sense
        new.raw = np.empty(new.raw.shape, dtype=new.dtype) # Copy values, breaking reference
        new.raw[new.auids] = arr
        return new

    @property
    def isnan(self):
        return self.asnew(np.isnan(self), cls=ss.BoolArr)