import numpy as np
import random
import util
from IPython import embed
import cv2 as cv
import time

class Particle(object):
    pos = (0,0)
    maxSpeed = 1.
    defaultColor = (50,50,255)
    color = defaultColor
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
        buffer = np.ceil(self.map.sampleResolution) 
        # Make sure we're within the map enough to take a sample TODO: replace 15 with sampleWidth/2
        return (min(max(pos_pixel[0], buffer), self.map.imageWidth-buffer), min(max(pos_pixel[1], buffer), self.map.imageHeight-buffer))


    def generateRandomMovementVector_map(self):
        r = random.random() * self.maxSpeed 
        randomSign = lambda: (-1 if random.random() < 0.5 else 1)
        x,y = (randomSign()*r, randomSign()*(self.maxSpeed - abs(r)) )

        if ( np.sqrt(x**2 + y**2) > 1 ):
            print("RANDOM MOVE TOO BIG")

        return x,y

    def move(self, dp_map):
        dx,dy = dp_map
        # dont let the agent wander off the map
        newPos = (self.pos[0] + dx, self.pos[1] + dy)

        newPos_pixel = self.map.positionToPixel(newPos)
        if (self.map.inBounds_pixel(newPos_pixel)):
            self.pos = newPos
        else:
            # print("Particle tried to move out of bounds. Halted.")
            pass

    def jostle(self):
        self.pos = (self.pos[0] + np.random.normal(0,0.03), self.pos[1] + np.random.normal(0,0.03))
        jostledPos = (self.pos[0] + np.random.normal(0,0.03), self.pos[1] + np.random.normal(0,0.03))
        jostled_pixel = self.map.positionToPixel(jostledPos)
        jostled_pixel = self.capPixelPosition(jostled_pixel)
        self.pos = self.map.pixelToPosition(jostled_pixel)
        

class ParticleFilter(object):
    particles = []
    dispersalStdev = 0.2

    def __init__(self, map, agent, numParticles=100, resampleRate=0.0):
        self.randomResample = resampleRate
        self.numParticles = numParticles
        self.particles = [Particle(map) for _ in range(self.numParticles)]
        self.map = map
        self.agent = agent


    def motionUpdate(self, dp):
        [p.move(dp) for p in self.particles]

    def comparisonFunction(self, measurement, expected_measurement):
        # dim = (8,8)
        # measurement = cv.resize(measurement, dim, interpolation = cv.INTER_AREA)
        # expected_measurement = cv.resize(expected_measurement, dim, interpolation = cv.INTER_AREA)

        compType = 3
        # Generate a histogram for each channel
        channels = [0,1,2]
        bins = 50
        a = cv.calcHist([measurement], [0,1,2], None, [bins,bins,bins], [0, 256, 0, 256, 0, 256])
        b = cv.calcHist([expected_measurement], [0,1,2], None, [bins,bins,bins], [0, 256, 0, 256, 0, 256])

        diff = cv.compareHist(a,b,compType)

        ret = (1 - diff)
        if (compType == 3):
            return ret if ret > 0.0001 else 0.0001
        else:
            return diff
        

    def calculateLikelihood(self):
        actual_measurement = self.map.sample(self.agent.pos)
        weights = [self.comparisonFunction(actual_measurement, self.map.sample(p.pos)) for p in self.particles]
        weightTotal = np.sum(weights)
        weights = [w/weightTotal for w in weights]
        p = list(zip(weights, self.particles))
        p.sort(reverse=True, key=lambda p: p[0]) # sort by probability

        return p

    def measurementUpdate(self):
        ps = self.calculateLikelihood()

        # for p in ps:
        #     print("{:01.2f} : {}".format(p[0],p[1].pos))

        # resample based on liklihood (move low probability particles to high prob locations)
        # take the cumulative sum of the weights for resampling
        weights = [pair[0] for pair in ps]
        cumulative_weights = np.cumsum(weights)

        sampledParticles = []
        # n^2 loop :(
        for i in range(int((1-self.randomResample) * self.numParticles)):
            rnd = random.random()
            for idx, entry in enumerate(cumulative_weights):
                if rnd < entry:
                    sampledP = Particle(self.map)
                    sampledP.pos = ps[idx][1].pos
                    sampledParticles.append(sampledP) # grab the particle
                    break


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
        # for i in range(int(0.1*self.numParticles)):
        while (len(self.particles) < self.numParticles):
            self.particles.append(Particle(self.map))
        
    def drawParticles(self, img=None):
        if img is None:
            img = self.map.image

        [util.drawCircle(img, self.map.positionToPixel(p.pos), p.color) for p in self.particles]

    # calc the distance of every particle from the ground truth
    def averageParticleDistance(self, groundTruth):
        # only care about the non-resamples particles
        distance = [util.distance(p.pos, groundTruth) for p in self.particles]
        distance = sorted(distance) # ascending order (closest->furthest)
        includedSamples = len(self.particles) * (1-self.randomResample)
        toIdx = int(np.ceil(includedSamples)-1)
        return np.mean(distance[0:toIdx])
    
    def numParticlesClusteredAroundGroundTruth(self, groundTruth):
        distance = [util.distance(p.pos, groundTruth) for p in self.particles]
        distance = [d for d in distance if d < 1]
        return len(distance)


    def fullUpdate(self, movementVector):
        self.motionUpdate(movementVector)
        self.measurementUpdate()
        self.drawParticles()
    


if __name__ == "__main__":
    pass