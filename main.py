import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import zplane as zp

def rotation():

    plt.gray()
    matriceImg = mpimg.imread('goldhill_rotate.png')

    #Application de la matrice de rotation
    matriceRota =  np.ndarray((512, 512, 4))

    xprime = []
    yprime = []
    for x in range(512):
        for y in range(512):
            xprime = x*0 + y*1
            yprime = x*-1 + y*0
            matriceRota[xprime][yprime] = matriceImg[x][y]

    mpimg.imsave("goldhill_transformed.png", matriceRota)

    return matriceRota

if __name__ == '__main__':
    rotation()