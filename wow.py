import numpy as np
import cv2
import pywt
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

class WOW:
    def __init__(self, image, payload, p=-1):
        self.image = image.astype(np.float32)
        self.payload = payload  
        self.p = p  
        self.sigma = 1.0  
        self.stego_image = np.copy(self.image)
        self.calculate_costs()

    def sobel_filters(self, img):
        """Применяет фильтры Собеля для вычисления градиентов."""
        sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_y = sobel_x.T
        Gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
        Gy = convolve2d(img, sobel_y, mode='same', boundary='symm')
        return Gx, Gy

    def wavelet_filters(self, img):
        """Выполняет вейвлет-разложение (Daubechies 8)."""
        coeffs = pywt.wavedec2(img, 'db8', level=1)
        LH, HL, HH = coeffs[1]  
        return LH, HL, HH

    def calculate_costs(self):
        """Рассчитывает стоимость изменения каждого пикселя на основе направленных фильтров."""
        Gx, Gy = self.sobel_filters(self.image)
        LH, HL, HH = self.wavelet_filters(self.image)
        residuals = [Gx, Gy, cv2.resize(LH, self.image.shape, interpolation=cv2.INTER_NEAREST),
                    cv2.resize(HL, self.image.shape, interpolation=cv2.INTER_NEAREST),
                    cv2.resize(HH, self.image.shape, interpolation=cv2.INTER_NEAREST)]
        
        self.costs = np.zeros_like(self.image)

        for res in residuals:
            safe_res = np.where(np.abs(res) < 1e-6, 1e-6, np.abs(res))  # Избегаем нулей
            self.costs += safe_res ** self.p

        self.costs = self.costs ** (-1 / self.p)
        self.costs = np.clip(self.costs, 1e-6, np.inf)


    

    def embed_message(self):
        """Встраивает сообщение в изображение, изменяя пиксели с наименьшей стоимостью."""
        num_pixels = int(self.payload * self.image.size)
        indices = np.argsort(self.costs.ravel())[:num_pixels]
        changes = np.random.choice([-1, 1], size=num_pixels)
        self.stego_image.ravel()[indices] += changes
        self.stego_image = np.clip(self.stego_image, 0, 255).astype(np.uint8)

    def get_stego_image(self):
        """Возвращает стеганографическое изображение."""
        return self.stego_image

#     @staticmethod
    def calculate_metrics(original, stego):
        """Вычисление метрик качества: PSNR, SSIM и MAD."""
        psnr = peak_signal_noise_ratio(original, stego, data_range=255)
        ssim = structural_similarity(original, stego, data_range=255)
        mad = np.mean(np.abs(original.astype(np.float32) - stego.astype(np.float32)))
        return psnr, ssim, mad

# Пример использования
if __name__ == '__main__':   
    name_cover_image = '123.jpg' 
    cover_image = cv2.imread(name_cover_image, cv2.IMREAD_GRAYSCALE)  
    if cover_image is None:
        print(f"Изображение '{name_cover_image}' не найдено. Генерируется случайно.")
        cover_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    payload = 0.25 
    wow = WOW(cover_image, payload)
    wow.embed_message()
    stego_image = wow.get_stego_image()
    
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cover_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(stego_image, cmap='gray')
    plt.title('Стегоизображение')
    plt.axis('off')
    
    
    plt.show(block=False)
    
    
    psnr, ssim, mad = WOW.calculate_metrics(cover_image, stego_image)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"MAD: {mad:.4f}")
    
    if name_cover_image == 'cover_image_wow.png':
        name_stego_image = 'stego_image_wow.png'
    else:
        if 'cover' == name_cover_image[:5]:
            name_stego_image = 'stego'+name_cover_image[5:]
        else:
            name_stego_image = 'stego_'+name_cover_image
    
    cv2.imwrite(f'{name_stego_image}', stego_image)
    print(f"Стегоизображение сохранено как '{name_stego_image}'")
