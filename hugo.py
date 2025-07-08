import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import matplotlib.pyplot as plt

def get_neighbors(image, i, j):
    """Возвращает значения 8-соседей пикселя (с учётом границ изображения)."""
    H, W = image.shape
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neighbors.append(image[ni, nj])
    return neighbors

def compute_local_cost(cover, i, j, candidate):
    """Вычисляет стоимость изменения пикселя с учетом соседей."""
    original_val = cover[i, j]
    neighbors = get_neighbors(cover, i, j)
    baseline = sum(abs(int(original_val) - int(n)) for n in neighbors)
    candidate_sum = sum(abs(int(candidate) - int(n)) for n in neighbors)
    return abs(candidate_sum - baseline)

def compute_complexity(cover):
    """Вычисляет карту сложности изображения, используя разностные фильтры."""
    H, W = cover.shape
    complexity = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            neighbors = get_neighbors(cover, i, j)
            complexity[i, j] = np.std(neighbors) 
    return complexity

def embed_hugo(cover, message, model_correction=True, seed=42):
    """Оптимизированный алгоритм встраивания по принципу HUGO с учётом текстуры."""
    stego = cover.copy().astype(np.int16)
    H, W = stego.shape
    complexity_map = compute_complexity(cover)
    indices = np.dstack(np.meshgrid(np.arange(H), np.arange(W))).reshape(-1, 2)
    np.random.default_rng(seed).shuffle(indices)
    if isinstance(message, str):
        message_bits = [int(bit) for bit in message]
    else:
        message_bits = list(message)
    msg_index, total_bits = 0, len(message_bits)
    for i, j in indices:
        if msg_index >= total_bits:
            break

        desired_bit = message_bits[msg_index]
        current_bit = stego[i, j] & 1
        if current_bit == desired_bit:
            msg_index += 1
            continue

        val = stego[i, j]
        candidate_up = val + 1 if val < 255 else None
        candidate_down = val - 1 if val > 0 else None

        cost_up = compute_local_cost(cover, i, j, candidate_up) if candidate_up is not None else np.inf
        cost_down = compute_local_cost(cover, i, j, candidate_down) if candidate_down is not None else np.inf

        if complexity_map[i, j] > np.median(complexity_map):
            if cost_up <= cost_down:
                stego[i, j] = candidate_up
            else:
                stego[i, j] = candidate_down
        msg_index += 1

    return np.clip(stego, 0, 255).astype(np.uint8)

def calculate_metrics(original, stego):
    """Вычисление метрик качества: PSNR, SSIM и MAD."""
    psnr = peak_signal_noise_ratio(original, stego, data_range=255)
    ssim = structural_similarity(original, stego, data_range=255)
    mad = np.mean(np.abs(original.astype(np.float32) - stego.astype(np.float32)))
    return psnr, ssim, mad

if __name__ == "__main__": 
    name_cover_image = '1231.jpg' #cover_high_hugo_2.jpg
    cover_image = cv2.imread(name_cover_image, cv2.IMREAD_GRAYSCALE)
    if cover_image is None:
        cover_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        
    np.random.seed(0)
    message_length = 60000
    random_message = ''.join(str(np.random.randint(0, 2)) for _ in range(message_length))
    stego_image = embed_hugo(cover_image, random_message, model_correction=True, seed=42)
    psnr, ssim, mad = calculate_metrics(cover_image, stego_image)
    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MAD:", mad)
    
    if name_cover_image == 'cover_image_hugo.png':
        name_stego_image = 'stego_image_hugo.png'
    else:
        if 'cover' == name_cover_image[:5]:
            name_stego_image = 'stego'+name_cover_image[5:]
        else:
            name_stego_image = 'stego_'+name_cover_image
    
    cv2.imwrite(name_cover_image, stego_image)
