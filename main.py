import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import zplane as zp


def lireImage():
    img = np.load("goldhill_bruit.npy")
    return img


def retirerAberrations(img):
    #Paramètres de la fonction de transfert inverse
    z_1 = 0
    z_2 = -0.99
    z_3 = 0.8
    z_4 = -0.99
    p_1 = 0.9 * pow(np.e, 1j * np.pi / 2)
    p_2 = 0.9 * pow(np.e, -1j * np.pi / 2)
    p_3 = 0.95 * pow(np.e, 1j * np.pi / 8)
    p_4 = 0.95 * pow(np.e, -1j * np.pi / 8)

    coeff_num = np.poly([z_1, z_2, z_3, z_4])
    coeff_denum = np.poly([p_1, p_2, p_3, p_4])

    #zp.zplane(coeff_num, coeff_denum)

    img = signal.lfilter(coeff_num, coeff_denum, img, axis=1)

    return img


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

    mpimg.imsave("goldhill_rotated.png", matriceRota)

    return matriceRota

def bruitFiltre(imageBruit):
    fe = 1600
    N1, Wn1 = signal.buttord(500, 750, 0.2, 60, fs=fe)
    print("Butterworth : " + str(N1))

    N2, Wn2 = signal.cheb1ord(500, 750, 0.2, 60, fs=fe)
    print("Chebyshev type I : " + str(N2))

    N3, Wn3 = signal.cheb2ord(500, 750, 0.2, 60, fs=fe)
    print("Chebyshev type II : " + str(N3))

    N4, Wn4 = signal.ellipord(500, 750, 0.2, 60, fs=fe)
    print("Elliptique : " + str(N4))

    b, a = signal.ellip(N4, 0.2, 60, Wn4, fs=fe)

    image = signal.lfilter(b, a, imageBruit)

    mpimg.imsave("goldhill_without_sound.png", image)


def bruitBilineaire(img):
    wc_d = 500 #Fréquence de coupure
    wc = 4789 #Fréquence de coupure selon gauchissement des fréquences
    fe = 1600 #Fréquence d'échantillonnage
    H_s = lambda s: 1 / (pow(s/wc, 2) + np.sqrt(2)*(s/wc) + 1)
    H_z = lambda z: H_s(2*fe*(z-1)/(z+1))

    coeff_num = [0.42, 0.84, 0.42]
    coeff_denum = [1, 0.46, 0.21]

    img = signal.lfilter(coeff_num, coeff_denum, img, axis=1)

    return img


def main():
    image = lireImage()
    #image = retirerAberrations(image)
    image = bruitBilineaire(image)
    mpimg.imsave("goldhill_transformed.png", image)

if __name__ == '__main__':
    plt.gray()
    main()

    img = np.load("goldhill_bruit.npy")
    bruitFiltre(img)
