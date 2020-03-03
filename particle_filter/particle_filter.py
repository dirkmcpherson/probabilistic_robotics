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
        # print("B- {:01.2f}, {:01.2f}".format(jostledPos[0],jostledPos[1]))
        jostled_pixel = self.map.positionToPixel(jostledPos)
        # print("C- {:01.2f}, {:01.2f}".format(jostledPos_pixel[0],jostledPos_pixel[1]))
        jostled_pixel = self.capPixelPosition(jostled_pixel)
        # print("D- {:01.2f}, {:01.2f}".format(jostledPos[0],jostledPos[1]))
        self.pos = self.map.pixelToPosition(jostled_pixel)
        

class ParticleFilter(object):
    totalExistingParticles = 0
    particles = []
    dispersalStdev = 0.2

    def __init__(self, map, agent, numParticles=100):
        self.numParticles = numParticles
        self.particles = [Particle(map) for _ in range(self.numParticles)]
        self.totalExistingParticles += len(self.particles)
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

        # all ones is best, all -1s is worst. -3 = 0, 3 = 1
        # for d in diffs:
        #     if d < 0:
        #         print("ERROR: FUNCTION IS NOT DOING WHAT YOU THINK")

        # totalDifference = sum(diffs) # move range to (0,6)

        # self.map.show(measurement, 'p')
        # self.map.show(expected_measurement, 'd')
        # print(1-diff)
        # cv.waitKey(0)

        ret = (1 - diff)
        if (compType == 3):
            return ret if ret > 0.0001 else 0.0001
        else:
            return diff

    # def comparisonFunction(self, measurement, expected_measurement):
    #     # straight up compare color (unbinned)
    #     maxDifference = 255 * 26 * 26 # A black image compared to a white image

    #     measurement = measurement.astype('int8')
    #     expected_measurement = expected_measurement.astype('int8')
    #     diffs = []
    #     channels = [0,1,2]
    #     for ch in channels:
    #         diffs.append(np.sum(np.abs(measurement[:,:,ch] - expected_measurement[:,:,ch])))


    #     # best case is diff is 0
    #     # diff = sum(diffs)
    #     # differenceRatio = diff / (3*maxDifference)

    #     diff = max(diffs)
    #     differenceRatio = diff / maxDifference

    #     return 1 - differenceRatio
        

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

        # embed()
        # time.sleep(10)
        weights = [self.comparisonFunction(actual_measurement, self.map.sample(p.pos)) for p in self.particles]
        weightTotal = np.sum(weights)
        print(weights)
        weights = [w/weightTotal for w in weights]
        p = list(zip(weights, self.particles))
        p.sort(reverse=True, key=lambda p: p[0]) # sort by probability

        return p

        # for entry in p:
        #     if entry[0] > 0.9:
        #         entry[1].color = (0,255,255)
        #     print(entry)


    randomResample = 0.0
    def measurementUpdate(self):
        ps = self.calculateLikelihood()

        for p in ps:
            print("{:01.2f} : {}".format(p[0],p[1].pos))

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

        # Fast but leaks particles
        # idx = 0
        # # d_sample = self.map.sample(self.agent.pos)
        # # self.map.show(d_sample,"agent")
        # while (len(sampledParticles) < 0.9*len(self.particles)):
        #     p = Particle(self.map)
        #     weight, particle = ps[idx]
        #     p.pos = particle.pos
        #     numTimesSampled = int(np.ceil(weight * len(self.particles)))
        #     # p_sample = self.map.sample(p.pos)
        #     # print("diff: ", self.comparisonFunction(d_sample, p_sample))
        #     # print("Weight: ", weight)
        #     # self.map.show(p_sample,"resample")
        #     # cv.waitKey(0)
        #     [sampledParticles.append(p) for i in range(numTimesSampled)]
        #     idx = idx + 1


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

    


if __name__ == "__main__":
    pass