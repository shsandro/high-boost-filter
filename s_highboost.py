import cv2

# carrega a imagem image
image = cv2.imread(input("Digite o caminho para a imagem de entrada: "))
# realiza borramento
gauss = cv2.GaussianBlur(image, (7, 7), 0)
# obtem a máscara
mask = cv2.subtract(image, gauss)
# obtem imagem final
final_image = cv2.add(image, mask)
cv2.imwrite(input("Digite o caminho para a imagem de saída: "), final_image)
