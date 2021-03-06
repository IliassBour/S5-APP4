### Auteurs:
###     Iliass Bourabaa (boui2215)
###     Pedro Maria Scoccimarro (scop2401)
###
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
    imgTourner = np.ndarray((x_size, y_size)) #512x512, 4

    matricePassage = np.array([[0, -1], [1, 0]])
    for x in range(x_size):
        for y in range(y_size):
            coordonner = np.array([[x], [y]])
            matriceRotation = np.matmul(matricePassage.T, coordonner)
            imgTourner[matriceRotation[0][0]][matriceRotation[1][0]] = img[x][y] #matriceImg

    plt.figure("Image avant rotation")
    plt.imshow(img, cmap="gray")
    plt.figure("Image après rotation")
    plt.imshow(imgTourner, cmap="gray")
    plt.show()

    return imgTourner

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
    fe = 1600 #fréquence d'échantillonnage
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
    matriceCov = np.cov(image) #matrice de covariance
    eigVal, eigVect = np.linalg.eig(matriceCov)
    imgComp = np.matmul(eigVect.T, image)

    #Compression de l'image
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

    imgCompresser50, eigVector50 = compression(image_bil, 50)
    imgDecompresser50 = decompression(imgCompresser50, eigVector50)
    imgCompresser70, eigVector70 = compression(image_bil, 70)
    imgDecompresser70 = decompression(imgCompresser70, eigVector70)

    plt.figure("Image débruitée avec méthode bilinéaire")
    plt.imshow(image_bil, cmap="gray")

    plt.figure("Image débruitée avec méthode python")
    plt.imshow(image_pyth, cmap="gray")

    plt.figure("Image décompréssée, 50%")
    plt.imshow(imgDecompresser50, cmap="gray")

    plt.figure("Image décompréssée, 70%")
    plt.imshow(imgDecompresser70, cmap="gray")

    plt.show()

if __name__ == '__main__':
    plt.gray()
    main()
