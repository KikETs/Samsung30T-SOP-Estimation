"""Freeze unidentifiable input channels to their exact training-time value."""
import copy
import numpy as np
from ecm_forward_v25 import Forward as BaseForward


class ConstantChannelScaler:
    def __init__(self,base,training):
        self.base=copy.deepcopy(base)
        z=base.transform(training)
        self.constant=(np.ptp(z,axis=0)==0)
        self.value=z[0].copy()
        self.medians=self.base.medians;self.mean=self.base.mean;self.std=self.base.std
        assert np.array_equal(self.transform(training),z)

    def transform(self,x):
        z=self.base.transform(x)
        z[:,self.constant]=self.value[self.constant]
        return z

    def unsupported_channels(self,x):
        z=self.base.transform(x)
        return (z!=self.value)&self.constant


class Forward(BaseForward):
    @classmethod
    def from_base(cls,base,training):
        obj=cls();obj.__dict__=copy.deepcopy(base.__dict__)
        obj.scaler=ConstantChannelScaler(base.scaler,training)
        obj.frozen_v25_weights=True
        return obj
