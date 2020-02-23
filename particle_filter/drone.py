#drone.py
import random
import numpy as np
from map import Map

class Drone(object):
    pos = (0,0)
    stdev = 0.2
    maxSpeed = 1.
    def __init__(self, map):
        self.map = map
        self.maxSpeed *= self.map.scale_percent
        self.stdev *= self.map.scale_percent
        # self.pos = (random.random() * map.imageWidth, random.random() * map.imageHeight)

    def generateRandomMovementVector(self):
        r = random.random() * self.maxSpeed 
        randomSign = lambda: (-1 if random.random() < 0.5 else 1)
        x,y = (randomSign()*r, randomSign()*(self.maxSpeed - abs(r)) )

        if ( np.sqrt(x**2 + y**2) > 1 ):
            print("RANDOM MOVE TOO BIG")

        return x,y

    def move(self, dp):
        dx,dy = dp
        nx = np.random.normal(0, self.stdev)
        ny = np.random.normal(0, self.stdev)

        # dont let the agent wander off the map
        newPos = (self.pos[0] + dx + nx, self.pos[1] + dy + ny)
        newPos_pixel = self.map.positionToPixel(newPos)
        if (self.map.inBounds(newPos_pixel)):
            self.pos = newPos
        else:
            print("Agent tried to move out of bounds. Halted.")

if __name__ == "__main__":
    m = Map(scale=25)
    d = Drone(m)

    m.showSample(d.pos)

