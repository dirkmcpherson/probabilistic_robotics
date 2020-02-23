import cv2 as cv
import numpy as np
import util
from IPython import embed

imagePath = "./MarioMap.png"

imageWidth = 0
imageHeight = 0
resolution = 50 # how many pixels per unit
def mapToImg(pos):
    global imageWidth, imageHeight
    # (0,0) is (w/2,h/2)
    x = (imageWidth/2) + resolution * pos[0]
    y = (imageHeight/2) + resolution * pos[1]
    return (x,y)

def main():
    img = cv.imread(imagePath, cv.IMREAD_COLOR)

    scaled_image = util.scaleImage(img, 25.)
    h,w,colorDim = scaled_image.shape

    global imageWidth, imageHeight
    imageWidth = w
    imageHeight = h

    for i in range(5):
        pos = mapToImg((i,0))
        print(pos)
        util.drawEllipse(scaled_image, pos)

    cv.imshow('image',scaled_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()