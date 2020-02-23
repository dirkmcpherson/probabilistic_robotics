import cv2 as cv
import numpy as np
import util
from IPython import embed
from map import Map
from drone import Drone 

imagePath = "./MarioMap.png"

def main():
    m = Map(imagePath, scale=25)
    d = Drone(m)

    finish = False
    sampleWidth = 216
    manualMoveIncrement = m.scale_percent
    while not finish:
        # m.sample_box(d.pos, sample_resolution=sampleWidth, draw=True)
        util.drawCircle(m.image, m.positionToPixel(d.pos))
        m.show()
        # cv.imshow('image', m.sample(d.pos, sample_resolution=sampleWidth))
        key = cv.waitKey(0)
        if (key == ord('q')):
            finish = True
        elif (key == ord('w')):
            d.move((0,manualMoveIncrement))
        elif (key == ord('s')):
            d.move((0,-manualMoveIncrement))
        elif (key == ord('a')):
            d.move((-manualMoveIncrement,0))
        elif (key == ord('d')):
            d.move((manualMoveIncrement,0))
        elif (key == 13): #enter
            dp = d.generateRandomMovementVector()
            d.move(dp)
        else:
            print("Unrecognized key")

        # box = m.sample_box(d.pos)
        # box = np.array(box, np.int32)
        # cv.polylines(m.image, [box], True, (255,255,255), 3)
        # Finish = True

if __name__ == "__main__":
    main()