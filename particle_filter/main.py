import cv2 as cv
import numpy as np
import util
from IPython import embed
from map import Map
from drone import Drone 
from particle_filter import Particle, ParticleFilter
import time
import util

imagePath = "./BayMap.png"
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
        # m.sample_box(d.pos, sample_resolution=sampleWidth, draw=True)

        drone_pos_pixel = m.positionToPixel(d.pos)
        drone_pos_map = m.pixelToPosition(drone_pos_pixel)
        util.drawCircle(m.image, m.positionToPixel(drone_pos_map))
        # util.drawCircle(m.image, m.positionToPixel(d.pos))
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

            # d_s = m.sample(d.pos)
            # m.show(d_s,'baseline')
            # cv.waitKey(0)
            # for i in range(4):
            #     xoffset = 0.05 * i
            #     for j in range(4):
            #         yoffset = 0.05*j

            #         p = Particle(m)
            #         p.pos = (d.pos[0]+xoffset, d.pos[1]+yoffset)
            #         p_s = m.sample(p.pos)
            #         print("Difference ", pf.comparisonFunction(d_s, p_s))
            #         m.show(p_s)
            #         cv.waitKey(0)


        # # if (update):
        # # t0 = time.time()
        # print("numP ", len(pf.particles))
        # # print("t0 ", time.time() - t0)
        # # print("t1 ", time.time() - t0)
        pf.motionUpdate(dp)
        # # print("t2 ", time.time() - t0)
        pf.measurementUpdate()
        # # print("t3 ", time.time() - t0)
        # # print("t4 ", time.time() - t0)
        pf.drawParticles()
            

        # box = m.sample_box(d.pos)
        # box = np.array(box, np.int32)
        # cv.polylines(m.image, [box], True, (255,255,255), 3)
        # Finish = True

if __name__ == "__main__":
    main(DEBUG=True)