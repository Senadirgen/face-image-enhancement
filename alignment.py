import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "dataset", "original")
output_path = os.path.join(BASE_DIR, "dataset", "aligned")
model_path = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")

OUTPUT_SIZE = (256, 256)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".pgm")

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(
        f"Dataset folder not found: {dataset_path}"
    )

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"LBF model not found: {model_path}"
    )

os.makedirs(output_path, exist_ok=True)

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Haar Cascade could not be loaded!")

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "Install opencv-contrib-python!"
    )

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(model_path)


def align_face(image):

    if image is None or image.size == 0:
        return None

    gray = image.copy()

    height, width = gray.shape

    equalized = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        equalized,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face_rectangle = np.array(
        [[x, y, w, h]],
        dtype=np.int32
    )

    try:
        success, landmarks = facemark.fit(
            gray,
            face_rectangle
        )
    except cv2.error:
        return None

    if not success or len(landmarks) == 0:
        return None

    points = np.asarray(landmarks[0]).reshape(-1, 2)

    if points.shape != (68, 2):
        return None

    if not np.all(np.isfinite(points)):
        return None

    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)

    if not np.all(np.isfinite(left_eye)):
        return None

    if not np.all(np.isfinite(right_eye)):
        return None

    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]

    if abs(delta_x) < 1:
        return None

    angle = np.degrees(
        np.arctan2(delta_y, delta_x)
    )

    if not np.isfinite(angle):
        return None

    eye_center = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2)
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        eye_center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        gray,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    face_center = np.array([
        x + w / 2,
        y + h / 2,
        1.0
    ])

    rotated_center = rotation_matrix @ face_center

    crop_size = int(max(w, h) * 1.5)
    crop_size = max(crop_size, 40)

    pad = crop_size

    padded = cv2.copyMakeBorder(
        rotated,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_REFLECT_101
    )

    crop_center = (
        float(rotated_center[0] + pad),
        float(rotated_center[1] + pad)
    )

    cropped = cv2.getRectSubPix(
        padded,
        (crop_size, crop_size),
        crop_center
    )

    if cropped is None or cropped.size == 0:
        return None

    aligned = cv2.resize(
        cropped,
        OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return aligned


total_images = 0
aligned_images = 0
failed_images = 0

for person_folder in sorted(os.listdir(dataset_path)):

    person_path = os.path.join(
        dataset_path,
        person_folder
    )

    if not os.path.isdir(person_path):
        continue

    person_output = os.path.join(
        output_path,
        person_folder
    )

    os.makedirs(person_output, exist_ok=True)

    image_count = 0

    for image_name in sorted(os.listdir(person_path)):

        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        total_images += 1

        image_path = os.path.join(
            person_path,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            failed_images += 1
            print(f"Cannot read: {image_path}")
            continue

        result = align_face(image)

        if result is None:
            failed_images += 1
            print(f"Alignment failed: {image_path}")
            continue

        output_file = os.path.join(
            person_output,
            os.path.splitext(image_name)[0] + ".png"
        )

        if cv2.imwrite(output_file, result):
            aligned_images += 1
            image_count += 1
        else:
            failed_images += 1
            print(f"Cannot save: {output_file}")

    print(
        f"{person_folder}: "
        f"{image_count} aligned images"
    )


print("\n================================")
print("       FACE ALIGNMENT RESULTS")
print("================================")

print(f"Total images: {total_images}")
print(f"Successfully aligned: {aligned_images}")
print(f"Failed images: {failed_images}")

print(
    f"Output size: "
    f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}"
)

print(f"Output folder: {output_path}")

print("================================")