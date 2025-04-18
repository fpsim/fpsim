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

    @property
    def isnan(self):
        return self.asnew(np.isnan(self), cls=ss.BoolArr)