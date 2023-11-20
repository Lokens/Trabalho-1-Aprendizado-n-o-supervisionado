import cv2
import os
import numpy as np
import re

valores_k = [2, 4, 6, 8, 10, 12, 14]
diretorio_imagens = "img"


def aplicar_kmeans(imagem, k):
    # Redimensionar a imagem para um vetor unidimensional
    pixels = imagem.reshape((-1, 3))
    pixels = np.float32(pixels)

    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Aplicar k-means
    _, etiquetas, centroides = cv2.kmeans(pixels, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converter os centroides de volta para o tipo de dados uint8
    centroides = np.uint8(centroides)

    # Mapea os pixels da imagem para os centroides
    imagem_segmentada = centroides[etiquetas.flatten()]

    # Remodelar a imagem de volta para o original
    imagem_segmentada = imagem_segmentada.reshape(imagem.shape)

    return imagem_segmentada


def print_info(nome_imagem):
    caminho_completo = os.path.join(diretorio_imagens, nome_imagem)
    imagem = cv2.imread(caminho_completo) # Carregar a imagem

    resolucao = imagem.shape[:2]
    tamanho_memoria_kb = os.path.getsize(caminho_completo) // 1024  # Tamanho ocupado em memória em KB
    cores_unicas = len(np.unique(imagem.reshape(-1, imagem.shape[2]), axis=0)) #Contagem de cores únicas em imagem

    resultado = re.search(r'_k(\d+)', nome_imagem)
    with open('informacoes.txt', 'a') as arquivo:
        if resultado : 
            arquivo.write(f"Informações para imagem {nome_imagem}:\n")
            arquivo.write(f"K={resultado.group(1)}:\n")
        else:
            arquivo.write(f"Informações para imagem {nome_imagem} (Original)\n")
            arquivo.write(f"K=Null\n")

        arquivo.write(f"Resolução: {resolucao}\n")
        arquivo.write(f"Tamanho em memória: {tamanho_memoria_kb} KB\n")
        arquivo.write(f"Cores únicas: {cores_unicas}\n\n\n")
    

for nome_imagem in os.listdir(diretorio_imagens):
    caminho_imagem = os.path.join(diretorio_imagens, nome_imagem)
   
    if os.path.isfile(caminho_imagem):
        imagem_original = cv2.imread(caminho_imagem) # Carregar a imagem
       
        # Aplicar o algoritmo k-means para cada valor de k
        for k in valores_k:
            imagem_segmentada = aplicar_kmeans(imagem_original, k)
            nome_saida = f"imagem_{nome_imagem.split('.')[0]}_k{k}.png"
            print(f"Gerando imagem {nome_saida}")

            # Salvar as imagens geradas
            cv2.imwrite(os.path.join(diretorio_imagens, nome_saida), imagem_segmentada)
            

for nome_imagem in os.listdir(diretorio_imagens):
    print(f"Salvando informações de {nome_imagem}")
    print_info(nome_imagem)
