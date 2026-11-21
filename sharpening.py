import os
import glob
import cv2
import numpy as np

# ==========================================
# 1. KESKİNLEŞTİRME FONKSİYONLARI (Modül 3)
# (4. Kişinin doğrudan import edip kullanabilmesi için modüler yapı)
# ==========================================

def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.2, threshold=0):
    """
    Unsharp Masking yöntemi ile göz ve dudak sınırlarını gürültüyü geri getirmeden netleştirir.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
        
    return sharpened

def laplacian_sharpening(image, alpha=0.3):
    """
    Laplacian filtresi ile kenar detaylarını belirginleştirir.
    alpha değeri optimum tutularak arka plan kumlanması önlenir.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    sharpened = image.astype(np.float64) - alpha * laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

# ==========================================
# 2. KESKİNLİK ÖLÇÜM METRİKLERİ
# ==========================================

def variance_of_laplacian(image):
    """
    Laplacian Varyansı: Değer arttıkça görselin keskinliği/netliği artmış demektir.
    """
    return float(cv2.Laplacian(image, cv2.CV_64F).var())

def tenengrad_metric(image):
    """
    Tenengrad Metriği: Sobel gradyan büyüklüklerinin karesi ortalaması ile kenar keskinliğini ölçer.
    """
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude_sq = gx**2 + gy**2
    return float(np.mean(grad_magnitude_sq))

# ==========================================
# 3. TOPLU İŞLEME VE SKOR KAYDI
# ==========================================

def process_dataset():
    # Göreli yollar (Hem senin bilgisayarında hem 4. kişide sorunsuz çalışır)
    input_folder = os.path.join("dataset", "denoised")
    output_unsharp_folder = os.path.join("dataset", "sharpened")
    output_laplacian_folder = os.path.join("dataset", "sharpened_laplacian")
    
    os.makedirs(output_unsharp_folder, exist_ok=True)
    os.makedirs(output_laplacian_folder, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.pgm')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))

    print(f"Denoised klasöründe bulunan fotoğraf sayısı: {len(image_paths)}")

    if len(image_paths) == 0:
        print("Hata: 'dataset/denoised' klasöründe fotoğraf bulunamadı! Klasör adını kontrol et.")
        return

    metrics = {
        "Denoised (Girdi)": {"lap_var": [], "tenengrad": []},
        "Unsharp Masking": {"lap_var": [], "tenengrad": []},
        "Laplacian Sharpening": {"lap_var": [], "tenengrad": []}
    }

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[Atlandı] Okunamadı: {img_path}")
            continue

        # Keskinleştirme algoritmalarını uygula
        img_unsharp = unsharp_masking(img, kernel_size=(5, 5), sigma=1.0, amount=1.2)
        img_laplacian = laplacian_sharpening(img, alpha=0.3)

        # Metrikleri hesapla
        metrics["Denoised (Girdi)"]["lap_var"].append(variance_of_laplacian(img))
        metrics["Denoised (Girdi)"]["tenengrad"].append(tenengrad_metric(img))

        metrics["Unsharp Masking"]["lap_var"].append(variance_of_laplacian(img_unsharp))
        metrics["Unsharp Masking"]["tenengrad"].append(tenengrad_metric(img_unsharp))

        metrics["Laplacian Sharpening"]["lap_var"].append(variance_of_laplacian(img_laplacian))
        metrics["Laplacian Sharpening"]["tenengrad"].append(tenengrad_metric(img_laplacian))

        # Görselleri kaydet (Unsharp Masking ana 'sharpened' klasörüne, Laplacian alternatif klasöre)
        base_name = os.path.basename(img_path)
        clean_name = base_name.replace("denoised_", "sharpened_")
        cv2.imwrite(os.path.join(output_unsharp_folder, clean_name), img_unsharp)
        cv2.imwrite(os.path.join(output_laplacian_folder, f"lap_{clean_name}"), img_laplacian)

    # Sonuçları hem terminale yazdır hem de txt dosyasına kaydet
    report_lines = ["=== DETAY KESKİNLEŞTİRME (MODÜL 3) ORTALAMA SKORLARI ===\n"]
    for method, vals in metrics.items():
        if len(vals["lap_var"]) > 0:
            avg_lap = np.mean(vals["lap_var"])
            avg_ten = np.mean(vals["tenengrad"])
            line = f"{method:22s} -> Ort. Laplacian Varyansı: {avg_lap:8.2f} | Ort. Tenengrad: {avg_ten:10.2f}"
            report_lines.append(line)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open("sharpening_score.txt", "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print("\nSkorlar 'sharpening_score.txt' dosyasına ve görseller 'dataset/sharpened' klasörüne kaydedildi!")

if __name__ == "__main__":
    process_dataset()