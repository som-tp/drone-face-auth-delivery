"""Step 3: the delivery simulation. The 'drone' lands, looks at the person,
and only releases the payload if it is the right recipient.

    python deliver.py --recipient alice
    python deliver.py --recipient alice --source test_video.mp4
"""
import argparse
import csv
import json
import time
from datetime import datetime

import cv2
import config
import face_utils
from gripper import Gripper


def log_event(recipient, result):
    new_file = not config.LOG_PATH.exists()
    with open(config.LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "recipient", "result"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), recipient, result])


def main():
    parser = argparse.ArgumentParser(description="Face-authenticated payload release")
    parser.add_argument("--recipient", required=True, help="who the parcel is for")
    parser.add_argument("--source", default=0, help="camera index or video file")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        raise SystemExit("No trained model. Run enroll.py then train.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(config.MODEL_PATH))
    id_to_name = {int(k): v for k, v in json.loads(config.LABELS_PATH.read_text()).items()}
    if args.recipient not in id_to_name.values():
        raise SystemExit(f"'{args.recipient}' is not enrolled. Known: {list(id_to_name.values())}")

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    gripper = Gripper()
    streak = 0            # consecutive frames where the right person was seen
    delivered = False
    last_denied_log = 0.0
    print(f"Waiting for recipient '{args.recipient}'. Press q to abort.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matched_this_frame = False

        for box in face_utils.detect_faces(gray):
            x, y, w, h = box
            face = face_utils.preprocess_face(gray, box)
            label_id, distance = recognizer.predict(face)    # lower distance = more similar
            name = id_to_name[label_id]

            if distance < args.threshold and name == args.recipient:
                matched_this_frame = True
                colour, text = (0, 255, 0), f"{name} ({distance:.0f})"
            elif distance < args.threshold:
                colour, text = (0, 165, 255), f"Not recipient: {name}"
            else:
                colour, text = (0, 0, 255), "Unknown person"
                if time.time() - last_denied_log > 5:      # log at most once every 5 s
                    log_event("unknown", "denied")
                    last_denied_log = time.time()

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        # Require several matches IN A ROW so one lucky frame can't unlock it.
        streak = streak + 1 if matched_this_frame else 0
        status = "PAYLOAD LOCKED"
        if streak >= config.FRAMES_REQUIRED and not delivered:
            gripper.open()
            log_event(args.recipient, "delivered")
            delivered = True
        if delivered:
            status = "PAYLOAD RELEASED"

        cv2.putText(frame, f"{status}  [{streak}/{config.FRAMES_REQUIRED}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Delivery", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if delivered:
            cv2.waitKey(2000)       # show the success screen for 2 seconds
            break

    gripper.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
