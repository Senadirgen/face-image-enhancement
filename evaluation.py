import os
import re
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 3. Modül (Detay Keskinleştirme) fonksiyonlarını doğrudan arkadaşının dosyasından alıyoruz
from sharpening import (
    unsharp_masking,
    laplacian_sharpening,
    variance_of_laplacian,
    tenengrad_metric,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGNED_DIR = os.path.join(BASE_DIR, "dataset", "aligned")


def parse_lighting_angle(filename):
    """
    Dosya isminden ışık açısını (Azimuth ve Elevation) okuyup
    ışığın önden ne kadar sapma yaptığını hesaplar.
    Örn: yaleB17_P00A-015E+20.png -> sqrt((-15)^2 + 20^2) = 25.0
    """
    match = re.search(r"A([+-]\d+)E([+-]\d+)", filename)
    if not match:
        return 0.0
    azimuth = int(match.group(1))
    elevation = int(match.group(2))
    return float(np.hypot(azimuth, elevation))


def load_dataset(aligned_dir=ALIGNED_DIR, train_ratio=0.6):
    """
    Hizalanmış yüz fotoğraflarını yükler ve her kişi için:
    - Daha normal/ön ışıklı fotoğrafları EĞİTİM (train) setine,
    - Daha zorlu/açılı ışıklı fotoğrafları TEST setine ayırır.
    """
    train_data = []
    test_data = []
    label_map = {}

    person_dirs = sorted(
        d for d in os.listdir(aligned_dir)
        if os.path.isdir(os.path.join(aligned_dir, d))
    )

    for label_id, person_name in enumerate(person_dirs):
        label_map[label_id] = person_name
        person_path = os.path.join(aligned_dir, person_name)

        person_items = []
        for fname in sorted(os.listdir(person_path)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".pgm")):
                continue
            fpath = os.path.join(person_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))
            angle = parse_lighting_angle(fname)
            person_items.append({
                "image": img,
                "label": label_id,
                "person": person_name,
                "filename": fname,
                "angle": angle,
            })

        # Işık açısına göre kolaydan zora sırala
        person_items.sort(key=lambda item: (item["angle"], item["filename"]))

        split_idx = int(len(person_items) * train_ratio)
        train_data.extend(person_items[:split_idx])
        test_data.extend(person_items[split_idx:])

    return train_data, test_data, label_map


def add_noise(image, noise_type="Gaussian", sigma=15, sp_amount=0.02, seed=42):
    """
    Sabit tohum (seed) kullanarak tekrarlanabilir yapay gürültü ekler.
    Böylece tüm filtreler birebir aynı gürültülü görüntü üzerinde kıyaslanır.
    """
    if noise_type == "None":
        return image.copy()

    rng = np.random.default_rng(seed)

    if noise_type == "Gaussian":
        gauss = rng.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy

    if noise_type == "Salt & Pepper":
        noisy = image.copy()
        num_salt = int(np.ceil(sp_amount * image.size * 0.5))
        salt_rows = rng.integers(0, image.shape[0], size=num_salt)
        salt_cols = rng.integers(0, image.shape[1], size=num_salt)
        noisy[salt_rows, salt_cols] = 255

        num_pepper = int(np.ceil(sp_amount * image.size * 0.5))
        pepper_rows = rng.integers(0, image.shape[0], size=num_pepper)
        pepper_cols = rng.integers(0, image.shape[1], size=num_pepper)
        noisy[pepper_rows, pepper_cols] = 0
        return noisy

    raise ValueError(f"Bilinmeyen gürültü türü: {noise_type}")


def apply_denoising(image, method="Bilateral"):
    """Modül 2: Klasik gürültü giderme filtreleri."""
    if method == "None":
        return image.copy()
    if method == "Gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    if method == "Median":
        return cv2.medianBlur(image, 3)
    if method == "Bilateral":
        return cv2.bilateralFilter(image, 5, 50, 50)
    if method == "NL-Means":
        return cv2.fastNlMeansDenoising(
            image, None, h=6, templateWindowSize=7, searchWindowSize=21
        )
    raise ValueError(f"Bilinmeyen filtre: {method}")


def apply_sharpening(image, method="Unsharp Masking"):
    """Modül 3: Detay keskinleştirme yöntemleri."""
    if method == "None":
        return image.copy()
    if method == "Unsharp Masking":
        return unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.2)
    if method == "Laplacian":
        return laplacian_sharpening(image, alpha=0.3)
    raise ValueError(f"Bilinmeyen keskinleştirme yöntemi: {method}")


def compute_metrics(reference_img, processed_img):
    """PSNR, SSIM, Laplacian Varyansı, Tenengrad ve Kontrast metriklerini hesaplar."""
    if np.array_equal(reference_img, processed_img):
        psnr_val = float("inf")
    else:
        psnr_val = float(psnr(reference_img, processed_img, data_range=255))

    return {
        "psnr": psnr_val,
        "ssim": float(ssim(reference_img, processed_img, data_range=255)),
        "lap_var": variance_of_laplacian(processed_img),
        "tenengrad": tenengrad_metric(processed_img),
        "contrast": float(np.std(processed_img)),
    }


def run_single_pipeline(
    image,
    noise_type="Gaussian",
    denoise_method="Bilateral",
    sharpen_method="Unsharp Masking",
    sigma=15,
    sp_amount=0.02,
    seed=42,
):
    """Tek bir fotoğrafı tüm aşamalardan geçirir ve her aşamanın metriklerini döndürür."""
    noisy = add_noise(image, noise_type=noise_type, sigma=sigma, sp_amount=sp_amount, seed=seed)
    denoised = apply_denoising(noisy, method=denoise_method)
    sharpened = apply_sharpening(denoised, method=sharpen_method)

    return {
        "original": image,
        "noisy": noisy,
        "denoised": denoised,
        "sharpened": sharpened,
        "metrics": {
            "original": compute_metrics(image, image),
            "noisy": compute_metrics(image, noisy),
            "denoised": compute_metrics(image, denoised),
            "sharpened": compute_metrics(image, sharpened),
        },
    }


def train_lbph(train_data):
    """Eğitim setiyle klasik LBPH yüz tanıma modelini eğitir."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = [item["image"] for item in train_data]
    labels = np.array([item["label"] for item in train_data], dtype=np.int32)
    recognizer.train(images, labels)
    return recognizer


def evaluate_scenario(
    recognizer,
    test_data,
    label_map,
    noise_type="Gaussian",
    denoise_method="None",
    sharpen_method="None",
    sigma=15,
    sp_amount=0.02,
):
    """Belirli bir yöntem kombinasyonunu tüm test setinde değerlendirir."""
    psnr_list, ssim_list, lap_list, ten_list, contrast_list = [], [], [], [], []
    records = []
    correct_count = 0

    for idx, item in enumerate(test_data):
        orig = item["image"]
        noisy = add_noise(
            orig, noise_type=noise_type, sigma=sigma, sp_amount=sp_amount, seed=42 + idx
        )
        denoised = apply_denoising(noisy, method=denoise_method)
        final_img = apply_sharpening(denoised, method=sharpen_method)

        m = compute_metrics(orig, final_img)
        if np.isfinite(m["psnr"]):
            psnr_list.append(m["psnr"])
        ssim_list.append(m["ssim"])
        lap_list.append(m["lap_var"])
        ten_list.append(m["tenengrad"])
        contrast_list.append(m["contrast"])

        pred_label, confidence = recognizer.predict(final_img)
        is_correct = int(pred_label == item["label"])
        correct_count += is_correct

        records.append({
            "filename": item["filename"],
            "angle": item["angle"],
            "true_person": label_map[item["label"]],
            "pred_person": label_map[pred_label],
            "correct": bool(is_correct),
            "confidence": float(confidence),
            "original": orig,
            "processed": final_img,
        })

    accuracy = (correct_count / len(test_data)) * 100.0 if test_data else 0.0
    summary = {
        "Avg PSNR (dB)": round(float(np.mean(psnr_list)), 2) if psnr_list else None,
        "Avg SSIM": round(float(np.mean(ssim_list)), 4),
        "Avg Laplacian Var": round(float(np.mean(lap_list)), 2),
        "Avg Tenengrad": round(float(np.mean(ten_list)), 2),
        "Avg Contrast": round(float(np.mean(contrast_list)), 2),
        "LBPH Accuracy (%)": round(accuracy, 2),
        "Correct / Total": f"{correct_count}/{len(test_data)}",
    }
    return summary, records


def run_full_benchmark(aligned_dir=ALIGNED_DIR, noise_type="Gaussian", sigma=15, sp_amount=0.02):
    """Tüm klasik yöntem kombinasyonlarını aynı test setinde kıyaslar."""
    train_data, test_data, label_map = load_dataset(aligned_dir=aligned_dir)
    recognizer = train_lbph(train_data)

    scenarios = [
        ("Original (Clean)", "None", "None", "None"),
        (f"Noisy ({noise_type})", noise_type, "None", "None"),
        ("Gaussian Filter", noise_type, "Gaussian", "None"),
        ("Median Filter", noise_type, "Median", "None"),
        ("Bilateral Filter", noise_type, "Bilateral", "None"),
        ("NL-Means Filter", noise_type, "NL-Means", "None"),
        ("Bilateral + Unsharp", noise_type, "Bilateral", "Unsharp Masking"),
        ("Bilateral + Laplacian", noise_type, "Bilateral", "Laplacian"),
        ("NL-Means + Unsharp", noise_type, "NL-Means", "Unsharp Masking"),
        ("NL-Means + Laplacian", noise_type, "NL-Means", "Laplacian"),
    ]

    rows = []
    details = {}

    for name, n_type, d_method, s_method in scenarios:
        summary, records = evaluate_scenario(
            recognizer,
            test_data,
            label_map,
            noise_type=n_type,
            denoise_method=d_method,
            sharpen_method=s_method,
            sigma=sigma,
            sp_amount=sp_amount,
        )
        rows.append({"Method": name, **summary})
        details[name] = records

    return pd.DataFrame(rows), details, (train_data, test_data, label_map)


if __name__ == "__main__":
    for noise in ["Gaussian", "Salt & Pepper"]:
        print(f"\n================ {noise.upper()} NOISE BENCHMARK ================")
        df, details, (train_d, test_d, _) = run_full_benchmark(noise_type=noise)
        print(f"Eğitim görsel sayısı: {len(train_d)} | Test görsel sayısı: {len(test_d)}\n")
        print(df.to_string(index=False))