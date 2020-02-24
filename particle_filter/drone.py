#drone.py
import random
import numpy as np
from map import Map
from particle_filter import Particle
from IPython import embed

class Drone(Particle):
    stdev = 0.2
    def __init__(self, map):
        super()
        self.map = map
        self.stdev *= self.map.scale_percent

    def move(self, dp):
        # udpate position without noise
        super(Drone,self).move(dp)

        # add noise
        nx = np.random.normal(0, self.stdev)
        ny = np.random.normal(0, self.stdev)

        newPos = (self.pos[0] + nx, self.pos[1] + ny)
        newPos_pixel = self.map.positionToPixel(newPos)
        if (self.map.inBounds(newPos_pixel)):
            self.pos = newPos
        else:
            print("Drone noise took it out of bounds. Ignored Noise.")

if __name__ == "__main__":
    m = Map(scale=25)
    d = Drone(m)

    m.showSample(d.pos)

