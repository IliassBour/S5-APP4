import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import zplane as zp


def lireImage():
    img = np.load("goldhill_aberrations.npy")
    return img


def retirerAberrations(img):
    #Paramètres de la fonction de transfert inverse
    z_1 = 0
    z_2 = -0.99
    z_3 = 0.8
    p_1 = 0.9 * pow(np.e, 1j * np.pi / 2)
    p_2 = 0.9 * pow(np.e, -1j * np.pi / 2)
    p_3 = 0.95 * pow(np.e, 1j * np.pi / 8)
    p_4 = 0.95 * pow(np.e, -1j * np.pi / 8)

    coeff_num = np.poly([z_1, z_2, z_3])
    coeff_denum = np.poly([p_1, p_2, p_3, p_4])

    #zp.zplane(coeff_num, coeff_denum)

    img = signal.lfilter(coeff_num, coeff_denum, img, axis=1)

    return img


def main():
    image = lireImage()
    image = retirerAberrations(image)
    mpimg.imsave("goldhill_transformed.png", image)

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
    plt.gray()
    main()
