import cv2
from skimage import io
import matplotlib.pylab as pylab

# carrega a imagem
image = cv2.imread(input("Digite o caminho para a imagem de entrada: "))[
    :, :, 0]

# lê a dimensão do kernel
n = int(input("Digite a dimensão da máscara: "))

# lê o valor de k
k = int(input("Digite o valor de k: "))

# realiza borramento
gauss = cv2.GaussianBlur(image, (n, n), 0)

# obtem a máscara (arestas)
mask = cv2.subtract(image, gauss)
mask = k * mask

####################### Mostra todos os passos da convolução #######################
pylab.figure(figsize=(30, 25))
pylab.gray()

pylab.subplot(2, 3, 1), pylab.imshow(image), pylab.title(
    'Imagem original', size=10), pylab.axis('off')

pylab.subplot(2, 3, 3), pylab.imshow(gauss)
pylab.title('Imagem após convolução', size=10), pylab.axis('off')

pylab.subplot(2, 3, 5), pylab.imshow(mask)
pylab.title('Arestas para aguçamento', size=10), pylab.axis('off')

pylab.subplots_adjust(wspace=0.2, hspace=1)
pylab.show()
##################################################################################################

# obtem imagem final
final_image = cv2.add(image, mask)
cv2.imwrite(input("Digite o caminho para a imagem de saída: "), final_image)
print("Imagem final salva!")
