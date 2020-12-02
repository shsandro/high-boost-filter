import scipy.fftpack as fp
from scipy import signal
import numpy as np
import numpy.fft
import cv2
import matplotlib.pylab as pylab


# carrega a imagem
img = cv2.imread(input("Digite o caminho para a imagem de entrada: "))[:, :, 0]

# lê o valor de D
D = int(input("Digite o valor de D para o kernel: "))

# obtém o filtro gaussiano no domínio espacial
gauss_kernel = np.outer(signal.gaussian(
    img.shape[0], D), signal.gaussian(img.shape[1], D))

# obtém a imagem e o filtro no domínio da frquência
freq_img = fp.fft2(img)
freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel))
assert(freq_img.shape == gauss_kernel.shape)

# aplica convolução (borramento)
convolved_img = freq_img * freq_kernel
blured_img = fp.ifft2(convolved_img).real

blured_img = np.abs(blured_img)
blured_img = blured_img * 255 / blured_img.max()
blured_img = blured_img.astype(np.uint8)

####################### Mostra todos os passos da convolução na frequência #######################
pylab.figure(figsize=(30, 25))
pylab.gray()

pylab.subplot(2, 2, 1), pylab.imshow(img), pylab.title(
    'Imagem original', size=10), pylab.axis('off')

pylab.subplot(2, 2, 4), pylab.imshow(blured_img)
pylab.title('Imagem após convolução', size=10), pylab.axis('off')

pylab.subplot(2, 2, 2), pylab.imshow(
    (20*np.log10(0.1 + fp.fftshift(freq_kernel))).astype(int))
pylab.title('Transformada do filtro', size=10), pylab.axis('off')

pylab.subplot(2, 2, 3), pylab.imshow(
    (20*np.log10(0.1 + fp.fftshift(convolved_img))).astype(int))
pylab.title('Imagem borrada no domínio da frequência',
            size=10), pylab.axis('off')

pylab.subplots_adjust(wspace=0.2, hspace=1)
pylab.show()
##################################################################################################

# lê o valor de k
k = int(input("Digite o valor de k: "))

# obtem a máscara (arestas)
mask = cv2.subtract(img, blured_img)
mask = k * mask

####################### Mostra todos os passos da convolução #######################
pylab.subplot(2, 3, 1), pylab.imshow(img), pylab.title(
    'Imagem original', size=10), pylab.axis('off')

pylab.subplot(2, 3, 3), pylab.imshow(blured_img)
pylab.title('Imagem após convolução', size=10), pylab.axis('off')

pylab.subplot(2, 3, 5), pylab.imshow(mask)
pylab.title('Arestas para aguçamento', size=10), pylab.axis('off')

pylab.subplots_adjust(wspace=0.2, hspace=1)
pylab.show()
##################################################################################################

# obtem imagem final
final_image = cv2.add(img, mask)
cv2.imwrite(input("Digite o caminho para a imagem de saída: "), final_image)
print("Imagem final salva!")
