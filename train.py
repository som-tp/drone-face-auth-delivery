"""Step 2: train the face recogniser on everything in dataset/.

    python train.py
"""
import json
import cv2
import numpy as np
import config


def load_dataset(dataset_dir=config.DATASET_DIR):
    """Return (images, label_ids, id_to_name) from dataset/<name>/*.jpg"""
    images, labels, id_to_name = [], [], {}
    people = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    for label_id, person in enumerate(people):
        id_to_name[label_id] = person.name
        for img_path in person.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_id)
    return images, labels, id_to_name


def build_recognizer():
    # LBPH = Local Binary Patterns Histograms (lives in opencv-contrib)
    return cv2.face.LBPHFaceRecognizer_create()


def main():
    images, labels, id_to_name = load_dataset()
    if not images:
        raise SystemExit("No images found. Run enroll.py first.")

    recognizer = build_recognizer()
    recognizer.train(images, np.array(labels))

    config.MODEL_DIR.mkdir(exist_ok=True)
    recognizer.write(str(config.MODEL_PATH))
    config.LABELS_PATH.write_text(json.dumps(id_to_name))
    print(f"Trained on {len(images)} images of {len(id_to_name)} people: "
          f"{list(id_to_name.values())}")


if __name__ == "__main__":
    main()
