import numpy as np
import random
import util
from IPython import embed
import cv2 as cv

class Particle(object):
    pos = (0,0)
    maxSpeed = 1.
    color = (50,50,255)
    def __init__(self, map):
        self.map = map
        self.maxSpeed *= self.map.scale_percent
        self.randomizePosition()


    def randomizePosition(self):
        # find a legal position and put it in map coordinates
        self.pos = (random.random() * self.map.imageWidth, random.random() * self.map.imageHeight)

        self.pos = self.capPixelPosition(self.pos)

        self.pos = self.map.pixelToPosition(self.pos)

    def capPixelPosition(self, pos_pixel):
        # Make sure we're within the map enough to take a sample TODO: replace 15 with sampleWidth/2
        return (min(max(pos_pixel[0], 15), self.map.imageWidth-15), min(max(pos_pixel[1], 15), self.map.imageHeight-15))


    def generateRandomMovementVector(self):
        r = random.random() * self.maxSpeed 
        randomSign = lambda: (-1 if random.random() < 0.5 else 1)
        x,y = (randomSign()*r, randomSign()*(self.maxSpeed - abs(r)) )

        if ( np.sqrt(x**2 + y**2) > 1 ):
            print("RANDOM MOVE TOO BIG")

        return x,y

    def move(self, dp):
        dx,dy = dp
        # dont let the agent wander off the map
        newPos = (self.pos[0] + dx, self.pos[1] + dy)

        newPos_pixel = self.map.positionToPixel(newPos)
        if (self.map.inBounds(newPos_pixel)):
            self.pos = newPos
        else:
            # print("Particle tried to move out of bounds. Halted.")
            pass

    def jostle(self):
        # print(self.pos)
        jostledPos = (self.pos[0] + np.random.normal(0,0.01), self.pos[1] + np.random.normal(0,0.01))
        jostledPos_pixel = self.capPixelPosition(self.map.positionToPixel(jostledPos))
        self.pos = self.map.pixelToPosition(jostledPos_pixel)
        

class ParticleFilter(object):
    particles = []
    dispersalStdev = 0.2

    def __init__(self, map, agent, numParticles=100):
        self.numParticles = numParticles
        self.particles = [Particle(map) for _ in range(self.numParticles)]
        self.map = map
        self.agent = agent

    def motionUpdate(self, dp):
        [p.move(dp) for p in self.particles]

    def comparisonFunction(self, measurement, expected_measurement):
        # Generate a histogram for each channel
        channels = [0,1,2]
        diffs = []
        for ch in channels:
            a = cv.calcHist([measurement], [ch], None, [256], [0,255])
            b = cv.calcHist([expected_measurement], [ch], None, [256], [0,255])
            diffs.append(cv.compareHist(a,b,3))



        # all ones is best, all -1s is worst. -3 = 0, 3 = 1

        for d in diffs:
            if d < 0:
                print("ERROR: FUNCTION IS NOT DOING WHAT YOU THINK")

        totalDifference = sum(diffs) # move range to (0,6)
        return totalDifference / 3.
        
        

    def calculateLikelihood(self):
        actual_measurement = self.map.sample(self.agent.pos)

        # for p in self.particles:
        #     expected_measurement = self.map.sample(p.pos)
        #     cv.imshow('actual', actual_measurement)
        #     cv.waitKey(0)
        #     cv.imshow('{}'.format(self.comparisonFunction(actual_measurement, expected_measurement)), expected_measurement)
        #     cv.waitKey(0)
        # return

        # weights = []
        # for p in self.particles:
        #     try:
        #         weights.append(self.comparisonFunction(actual_measurement, self.map.sample(p.pos)))
        #     except:
        #         embed()
        #         return
        weights = [self.comparisonFunction(actual_measurement, self.map.sample(p.pos)) for p in self.particles]
        weightTotal = sum(weights)
        weights = [w/weightTotal for w in weights]

        p = list(zip(weights, self.particles))
        p.sort(key=lambda p: p[0]) # sort by probability

        return p

        # for entry in p:
        #     if entry[0] > 0.9:
        #         entry[1].color = (0,255,255)
        #     print(entry)


    def measurementUpdate(self):
        ps = self.calculateLikelihood()

        # resample based on liklihood (move low probability particles to high prob locations)
        # take the cumulative sum of the weights for resampling
        weights = [pair[0] for pair in ps]
        cumulative_weights = np.cumsum(weights)

        sampledParticles = []
        # n^2 loop :(
        for i in range(int(0.9 * self.numParticles)):
            for idx, entry in enumerate(cumulative_weights):
                if random.random() < entry:
                    sampledParticles.append(ps[idx][1]) # grab the particle

        self.particles = sampledParticles
        '''
        In order to resample we take the cumulative sum of the weights array ordered from the particle with the lowest weight to the particle with the most weight. Then we grab a random number and sample whichever particle is the closest to that number without going over. This resamples proportionally to weights, but prevents sample impoverishment of the kind we'd get if we just direcly sampled a particle in proportion to its weight. 
        '''

        # noise every particle a little bit
        [p.jostle() for p in self.particles]
        
        '''
        Once 90% particles are sampled and jostled with a small amount of univariate gaussian noise, an additional 10% of the particles 
        are randomly distributed over the map. This allows the filter to find the true position of the robot again if the clusters dont form well. 
        '''

        # now randomly disperse 10% of particles
        for i in range(int(0.1*self.numParticles)):
            self.particles.append(Particle(self.map))
        
    def drawParticles(self, img=None):
        if img is None:
            img = self.map.image

        [util.drawCircle(img, self.map.positionToPixel(p.pos), p.color) for p in self.particles]

    


if __name__ == "__main__":
    pass