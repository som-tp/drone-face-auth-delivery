"""Step 1: capture face photos of an authorised recipient.

    python enroll.py --name alice
"""
import argparse
import cv2
import config
import face_utils


def main():
    parser = argparse.ArgumentParser(description="Enroll a new recipient")
    parser.add_argument("--name", required=True, help="recipient name (no spaces)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = config.DATASET_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    saved, frame_no = 0, 0
    print("Look at the camera and slowly move your head a little. Press q to quit.")
    while saved < config.SAMPLES_PER_PERSON:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_utils.detect_faces(gray)
        frame_no += 1

        for box in faces[:1]:                       # only use the first face
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if frame_no % 5 == 0:                   # save every 5th frame -> more variety
                face = face_utils.preprocess_face(gray, box)
                cv2.imwrite(str(out_dir / f"{saved:03d}.jpg"), face)
                saved += 1

        cv2.putText(frame, f"Saved {saved}/{config.SAMPLES_PER_PERSON}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Enroll", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved} images saved to {out_dir}")


if __name__ == "__main__":
    main()
