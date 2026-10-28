"""Optional: measure accuracy so you have a real number for your resume.

Needs at least 2 enrolled people. Splits each person's photos 80/20,
trains on 80%, tests on the unseen 20%.

    python evaluate.py
"""
import random
import numpy as np
import config
from train import load_dataset, build_recognizer


def main(seed=42):
    images, labels, id_to_name = load_dataset()
    if len(id_to_name) < 2:
        raise SystemExit("Enroll at least 2 people to evaluate.")

    random.seed(seed)
    idx = list(range(len(images)))
    random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    rec = build_recognizer()
    rec.train([images[i] for i in train_idx], np.array([labels[i] for i in train_idx]))

    correct = 0
    for i in test_idx:
        pred, _ = rec.predict(images[i])
        correct += int(pred == labels[i])
    print(f"Test accuracy: {correct}/{len(test_idx)} = {100 * correct / len(test_idx):.1f}%")


if __name__ == "__main__":
    main()
