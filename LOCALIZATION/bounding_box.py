import math
from point import Point

class BoundingBox(object):
    def __init__(self, pmin, pmax):

        self._pmin = pmin
        self._pmax = pmax
        pcenter = Point(-1,-1)
        pcenter.x = self._pmin.x + int((self._pmax.x - self._pmin.x) / 2)
        pcenter.y = self._pmin.y + int((self._pmax.y - self._pmin.y) / 2)
        self._pcenter = pcenter
    
    @property
    def center(self):
        return self._pcenter
    
    @property
    def pmin(self): 
        return self._pmin 
    @pmin.setter
    def pmin(self, pmin):
        self._pmin = pmin

    @property
    def pmax(self): 
        return self._pmax
    @pmax.setter
    def pmax(self, pmax):
        self._pmax = pmax

    def area(self):
        dy = self._pmax.y - self._pmin.y
        dx = self._pmax.x - self._pmin.x
        return dx * dy 

    
    