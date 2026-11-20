import os
import glob
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. Adım: 1. kişinin ürettiği ANA KLASÖRÜN yolunu buraya yaz
INPUT_FOLDER = r'C:\Users\enes\Documents\GitHub\face-image-enhancement\dataset\aligned'

# İstersen işlenen temiz görselleri kaydedeceğin klasör yolu:
OUTPUT_FOLDER = r'C:\Users\enes\Documents\GitHub\face-image-enhancement\dataset\denoised'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Klasör ve tüm alt klasörlerdeki görselleri topla
extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.pgm')
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, '**', ext), recursive=True))

print(f"Toplam bulunan fotoğraf sayısı: {len(image_paths)}")

if len(image_paths) == 0:
    print("Hata: Belirtilen klasörde veya alt klasörlerinde hiç fotoğraf bulunamadı! Yolu kontrol et.")
else:
    # Tüm fotoğraflar için skorları biriktireceğimiz listeler
    scores = {
        "Gaussian": {"psnr": [], "ssim": []},
        "Bilateral": {"psnr": [], "ssim": []},
        "NL_Means": {"psnr": [], "ssim": []},
        "Median_SP": {"psnr": [], "ssim": []},
    }

    for idx, img_path in enumerate(image_paths):
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Bozuk veya okunamayan dosya varsa kodu durdurmadan atla
        if original_image is None:
            print(f"[Atlandı] Okunamadı: {img_path}")
            continue

        original_image = cv2.resize(original_image, (256, 256))

        # 2. Adım: Yapay Gürültü Ekleme
        row, col = original_image.shape
        gauss = np.random.normal(0, 15, (row, col)).astype(np.float32)
        gaussian_noisy = np.clip(original_image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        salt_pepper_noisy = original_image.copy()
        num_salt = np.ceil(0.02 * original_image.size * 0.5)
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in original_image.shape]
        salt_pepper_noisy[tuple(coords_salt)] = 255

        num_pepper = np.ceil(0.02 * original_image.size * 0.5)
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in original_image.shape]
        salt_pepper_noisy[tuple(coords_pepper)] = 0

        # 3. Adım: Filtreleri Uygulama
        img_gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (3, 3), 0)
        img_median_filtered = cv2.medianBlur(salt_pepper_noisy, 3)
        img_bilateral = cv2.bilateralFilter(gaussian_noisy, 5, 50, 50)
        img_nl_means = cv2.fastNlMeansDenoising(gaussian_noisy, None, h=6, templateWindowSize=7, searchWindowSize=21)

        # 4. Adım: Metrikleri Hesapla ve Kaydet
        scores["Gaussian"]["psnr"].append(psnr(original_image, img_gaussian_filtered))
        scores["Gaussian"]["ssim"].append(ssim(original_image, img_gaussian_filtered))

        scores["Bilateral"]["psnr"].append(psnr(original_image, img_bilateral))
        scores["Bilateral"]["ssim"].append(ssim(original_image, img_bilateral))

        scores["NL_Means"]["psnr"].append(psnr(original_image, img_nl_means))
        scores["NL_Means"]["ssim"].append(ssim(original_image, img_nl_means))

        scores["Median_SP"]["psnr"].append(psnr(original_image, img_median_filtered))
        scores["Median_SP"]["ssim"].append(ssim(original_image, img_median_filtered))

        # İşlenen görseli (örneğin Bilateral çıktısını) kaydet
        base_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_FOLDER, f"denoised_{idx}_{base_name}")
        cv2.imwrite(save_path, img_bilateral)

    # 5. Adım: Tüm verisetindeki ortalama skorları yazdır
    print("\n=== TÜM GÖRSELLER İÇİN ORTALAMA SONUÇLAR ===")
    for method, vals in scores.items():
        if len(vals["psnr"]) > 0:
            avg_p = np.mean(vals["psnr"])
            avg_s = np.mean(vals["ssim"])
            print(f"{method:12s} -> Ortalama PSNR: {avg_p:.2f} dB | Ortalama SSIM: {avg_s:.4f}")