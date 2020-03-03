import cv2 as cv
import numpy as np
import util
from IPython import embed
from map import Map
from drone import Drone 
from particle_filter import Particle, ParticleFilter
import time
import util
import matplotlib.pyplot as plt

imagePath = "./CityMap.png" #
imagePath ="./MarioMap.png" #"./BayMap.png"
def main(DEBUG=False):
    m = Map(imagePath, scale=100, sampleResolution=26)
    d = Drone(m)
    pf = ParticleFilter(m, d, numParticles=1000)

    pf.calculateLikelihood()
    pf.drawParticles()

    # embed()

    finish = False
    manualMoveIncrement = m.scale_percent
    while not finish:
        update=False
        dp = (0,0)
        util.drawCircle(m.image, m.positionToPixel(d.pos))
        m.show()
        m.clearImage()
        # cv.imshow('image', m.sample(d.pos, sample_resolution=sampleWidth))
        key = cv.waitKey(0)
        if (key == ord('q')):
            finish = True
        elif (key == ord('w')):
            dp = (0,-manualMoveIncrement)
            d.move(dp)
            update=True
        elif (key == ord('s')):
            dp = (0,manualMoveIncrement)
            d.move(dp)
            update=True
        elif (key == ord('a')):
            dp = (-manualMoveIncrement,0)
            d.move(dp)
            update=True
        elif (key == ord('d')):
            dp = (manualMoveIncrement,0)
            d.move(dp)
            update=True
        elif (key == 13): #enter
            dp = d.generateRandomMovementVector_map()
            d.move(dp)
            update=True
        else:
            print("Unrecognized key")

        if (DEBUG):
            distance = sum([util.distance(p.pos, d.pos) for p in pf.particles])/len(pf.particles)
            print("true: ",d.pos)
            [print("{:01.2f} : {:01.2f}".format(p.pos[0],p.pos[1])) for p in pf.particles]
            print("distance: ", distance)

        pf.fullUpdate(dp)

# Basic run
def generateDataPoint1(nP, resampleRate, numIterations, headless):
    ret = []
    m = Map(imagePath, scale=100, sampleResolution=26)
    d = Drone(m)
    pf = ParticleFilter(m, d, numParticles=nP, resampleRate=resampleRate)
    pf.calculateLikelihood()

    for i in range(numIterations):
        dp = d.generateRandomMovementVector_map()
        d.move(dp)
        pf.fullUpdate(dp)

        # ret.append(pf.averageParticleDistance(d.pos))
        ret.append(pf.numParticlesClusteredAroundGroundTruth(d.pos))

        if not headless:
            m.clearImage()
            pf.drawParticles()
            util.drawCircle(m.image, m.positionToPixel(d.pos))
            m.show()
            key = cv.waitKey(0)

    return ret


def runExperiment1(numParticles=1000, experimentalRuns=5, iterationsPerRun=10, resampleRate=0.0, headless=True):
    y = []
    for i in range(experimentalRuns):
        print("Run ", i)
        y.append(generateDataPoint1(numParticles, resampleRate, iterationsPerRun, headless))


    # average the results of each run
    y = np.array(y)
    y = np.mean(y, 0)

    x = [i for i in range(len(y))]
    # plt.plot(x,y)
    # plt.title("Particle distance from Drone over time. Averaged {} runs with {} particles.".format(experimentalRuns, numParticles))
    # plt.xlabel("Particle Filter Update")
    # plt.ylabel("Average Distance between Particle and Drone")
    # plt.show()

    return x,y

if __name__ == "__main__":
    # main(DEBUG=True) 
    runs=20
    iterations=25
    numParticles = 1000
    x,y = runExperiment1(numParticles, runs, iterations, resampleRate=0.0, headless=True)
    x0,y0 = runExperiment1(numParticles, runs, iterations, resampleRate=0.1, headless=True)
    x1,y1 = runExperiment1(numParticles, runs, iterations, resampleRate=0.2, headless=True)

    plt.plot(x,y,x0,y0,x1,y1)
    plt.legend(["No resample","10% resample", "20% resample"])
    plt.xlabel("Particle Filter Update")
    plt.title("Particles Close to Drone. Averaged {} runs with {} particles.".format(runs, numParticles))
    plt.ylabel("Particles within 1 unit of Drone")
    # plt.title("Particle distance from Drone over time. Averaged {} runs with {} particles.".format(runs, numParticles))
    # plt.ylabel("Average Distance between Particle and Drone")
    plt.show()