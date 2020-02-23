
import cv2 as cv
import util
import numpy as np
from IPython import embed

class Map(object):
    resolution = 50 #pixels per distance unit
    scale_percent = 30.
    def __init__(self, img_path="./MarioMap.png", scale=100):
        self.scale_percent = scale / 100.
        self.original_image = cv.imread(img_path, cv.IMREAD_COLOR)
        self.image = util.scaleImage(self.original_image, scale)

        rows,cols,colorDim = self.image.shape
        self.imageWidth = cols
        self.imageHeight = rows

    def pixelToPosition(self, pos_pixel):
        x = (pos_pixel[0] - self.imageWidth/2) / self.resolution
        y = (pos_pixel[1] - self.imageHeight/2) / self.resolution
        return (x,y)

    def positionToPixel(self, pos_position):
        # (0,0) is (w/2,h/2)
        x = (self.imageWidth/2) + self.resolution * pos_position[0]
        y = (self.imageHeight/2) - self.resolution * pos_position[1]
        return (x,y)

    def inBounds(self, pos_pixel):
        x,y = pos_pixel
        return (x > 0 and y > 0 and x < self.imageWidth and y < self.imageHeight)

    def sample_box(self, position, sample_resolution=26, draw=False):
        x,y = self.positionToPixel(position)
        minx, maxx = (int(x - sample_resolution / 2),  int(x + sample_resolution / 2))
        miny, maxy = (int(y - sample_resolution / 2),  int(y + sample_resolution / 2))

        box = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]

        if (draw):
            box = np.array(box, np.int32)
            cv.polylines(self.image, [box], True, (255,255,255), 3)

        return box


    def sample(self, position, sample_resolution=128): # make sample resolution even ALWAYS
        x,y = self.positionToPixel(position)
        box = self.sample_box(position, sample_resolution=sample_resolution)
        
        minx = box[0][0]
        miny = box[0][1]
        subset = []
        if not self.inBounds((x,y)):
            # Literal Edge case
            print("Tried to sample out of bounds. Not allowed (for now)")
            pass
        else:
            # embed()
            subset = self.image[miny:miny+int(sample_resolution), minx:minx+sample_resolution]
        
        print(subset)
        return subset

    def showSample(self, pixel_position):
        self.show(self.sample(pixel_position))

    def show(self, img=None):
        if img is None:
            img = self.image

        # # scaled_image = util.scaleImage(self.image, 25.)
        cv.imshow('image',img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def mark(self, pos_pixel, color='red'):
        pass

if __name__ == "__main__":
    imagePath = "./MarioMap.png"
    map = Map(imagePath, 25) 

    for i in range(5):
        pos = map.positionToPixel((i,0))
        util.drawEllipse(map.image, pos)
    map.show()