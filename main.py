### Auteurs:
###     Iliass Bourabaa (boui2215)
###     Pedro Maria Scoccimarro (scop2401)
###
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import zplane as zp


def lireImage(img_name):
    img = np.load(img_name)
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

    plt.title("Pôles et Zéros pour abérrations")
    zp.zplane(coeff_num, coeff_denum)

    plt.figure("Image avec abérrations")
    plt.imshow(img, cmap="gray")

    img = signal.lfilter(coeff_num, coeff_denum, img, axis=1)

    plt.figure("Image sans abérrations")
    plt.imshow(img, cmap="gray")
    plt.show()

    return img


def rotation(img):
    #Application de la matrice de rotation
    x_size = len(img)
    y_size = len(img[0])
    matriceRota = np.ndarray((x_size, y_size)) #512x512, 4

    for x in range(x_size):
        for y in range(y_size):
            xprime = x*0 + y*1
            yprime = x*-1 + y*0
            matriceRota[xprime][yprime] = img[x][y] #matriceImg

    plt.figure("Image avant rotation")
    plt.imshow(img, cmap="gray")
    plt.figure("Image après rotation")
    plt.imshow(matriceRota, cmap="gray")
    plt.show()

    return matriceRota

def bruitBilineaire(img):
    wc_d = 500 #Fréquence de coupure
    wc = 4789 #Fréquence de coupure selon gauchissement des fréquences
    fe = 1600 #Fréquence d'échantillonnage
    H_s = lambda s: 1 / (pow(s/wc, 2) + np.sqrt(2)*(s/wc) + 1)
    H_z = lambda z: H_s(2*fe*(z-1)/(z+1))

    coeff_num = [0.42, 0.84, 0.42]
    coeff_denum = [1, 0.46, 0.21]

    img = signal.lfilter(coeff_num, coeff_denum, img, axis=1)

    plt.title("Pôles et Zéros méthode bilinéaire")
    zp.zplane(coeff_num, coeff_denum)

    w, h = signal.freqz(coeff_num, coeff_denum)
    plt.figure("Reponse fréquentielle bilinéaire")
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.ylabel("Fréquence (dB)")
    plt.xlabel("w")
    plt.show()

    return img

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

    coeff_num, coeff_denum = signal.ellip(N4, 0.2, 60, Wn4, fs=fe)

    image = signal.lfilter(coeff_num, coeff_denum, imageBruit)

    plt.title("Pôles et Zéros méthode python (Elliptique)")
    zp.zplane(coeff_num, coeff_denum)

    w, h = signal.freqz(coeff_num, coeff_denum)
    plt.figure("Reponse fréquentielle bilinéaire")
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.ylabel("Fréquence (dB)")
    plt.xlabel("w")
    plt.show()

    return image

def compression(image, pourcentage):
    #matrice
    matriceCov = np.cov(image)

    #eigne value et vector
    eigVal, eigVect = np.linalg.eig(matriceCov)

    # matriceComp = eigVect*image
    imgComp = np.matmul(eigVect.T, image)

    nbLigne = int(len(imgComp)*(100-pourcentage)/100)

    for x in range(len(imgComp)):
         if x >= nbLigne:
            for y in range(len(imgComp[0])):
                imgComp[x][y] = 0

    return imgComp, eigVect

def decompression(image, eigVector):
    invEigVect = np.linalg.inv(eigVector)

    imgDecomp = np.matmul(invEigVect.T, image)

    return imgDecomp



def main():
    image = lireImage("image_complete.npy")
    image = retirerAberrations(image)
    image = rotation(image)
    image_bil = bruitBilineaire(image)
    image_pyth = bruitFiltre(image)

    plt.figure("Image débruitée avec méthode bilinéaire")
    plt.imshow(image_bil, cmap="gray")

    plt.figure("Image débruitée avec méthode python")
    plt.imshow(image_pyth, cmap="gray")
    plt.show()

    imgCompresser, eigVector = compression(image, 75)
    imgDecompresser = decompression(imgCompresser, eigVector)
    mpimg.imsave("chaton_decomp.png", imgDecompresser)

if __name__ == '__main__':
    plt.gray()
    main()
