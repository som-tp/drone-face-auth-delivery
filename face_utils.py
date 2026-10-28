"""Small helpers shared by every script."""
import cv2
import config

# Haar cascade ships with OpenCV, so nothing extra to download.
_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(gray):
    """Return a list of (x, y, w, h) boxes for faces found in a grayscale frame."""
    return _detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )


def preprocess_face(gray, box):
    """Crop the face, resize it, and even out the lighting.

    equalizeHist spreads brightness values out, which makes the same face
    look more similar under dim vs. bright light.
    """
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)
    return cv2.equalizeHist(face)
