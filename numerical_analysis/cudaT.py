import numpy as np
from numba import cuda
import math

# Define a função que rodará na GPU
@cuda.jit
def render_kernel(imagem_saida):
    # Pega as coordenadas (x, y) que esta thread específica vai cuidar
    x, y = cuda.grid(2)

    # Define as dimensões
    largura = 1920
    altura = 1080
    
    # Garante que a thread está dentro dos limites da imagem
    if x < largura and y < altura:
        
        # 1. Calcular o índice linear do pixel
        # (usamos 'long' para evitar estouro de números grandes)
        indice_pixel = np.int64(y * largura + x)

        # 2. Definir os totais
        total_pixels = np.int64(largura * altura)
        total_cores = np.int64(256 * 256 * 256)

        # 3. Mapear o índice do pixel para o espaço de cores
        # Esta é a interpolação principal:
        valor_cor_24bit = (indice_pixel * (total_cores - 1)) // (total_pixels - 1)

        # 4. Descompactar a cor 24-bit em R, G, B (8-bit cada)
        # (x >> 16) & 0xFF  -> Pega os 8 bits mais à esquerda (Vermelho)
        # (x >> 8)  & 0xFF  -> Pega os 8 bits do meio (Verde)
        # x         & 0xFF  -> Pega os 8 bits mais à direita (Azul)
        
        r = (valor_cor_24bit >> 16) & 0xFF
        g = (valor_cor_24bit >> 8)  & 0xFF
        b =  valor_cor_24bit        & 0xFF

        # 5. Escrever o resultado na matriz de saída
        imagem_saida[y, x, 0] = r
        imagem_saida[y, x, 1] = g
        imagem_saida[y, x, 2] = b

# --- Código que roda no CPU (Host) ---

# 1. Define as dimensões
largura = 1920
altura = 1080

# 2. Aloca a memória para a imagem no CPU
# Usamos np.uint8 pois as cores vão de 0 a 255
imagem_host = np.zeros((altura, largura, 3), dtype=np.uint8)

# 3. Aloca memória na GPU e copia os dados (nesse caso, uma matriz de zeros)
imagem_device = cuda.to_device(imagem_host)

# 4. Define como vamos dividir o trabalho
# (Threads por bloco e blocos no grid)
threads_por_bloco = (16, 16)  # Blocos de 16x16 threads
blocos_x = math.ceil(largura / threads_por_bloco[0])
blocos_y = math.ceil(altura / threads_por_bloco[1])
blocos_no_grid = (blocos_x, blocos_y)

# 5. Lança o Kernel!
render_kernel[blocos_no_grid, threads_por_bloco](imagem_device)

# 6. Copia o resultado da GPU de volta para o CPU
imagem_host = imagem_device.copy_to_host()

# 7. Agora 'imagem_host' contém sua imagem!
# (Você pode salvá-la com PIL/Pillow)
from PIL import Image
img = Image.fromarray(imagem_host, 'RGB')
img.save('gradiente_cuda.png')