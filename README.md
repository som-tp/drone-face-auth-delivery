# Drone Payload Delivery – Face Authentication (OpenCV)

A Python/OpenCV simulation of the security part of a delivery drone: the drone only
**releases its payload when the correct recipient's face is recognised**.

This is the software side of my university team project *"Drone Payload Delivery System
Using Facial Recognition"* (quadcopter with Pixhawk 2 autopilot, Raspberry Pi + camera,
servo gripper). This repo is my re-implementation of the face-authentication module so
it can be run on any laptop with a webcam — no drone needed.

## How it works

```
Camera frame
    │
    ▼
Detect face (Haar Cascade) ──► no face? keep waiting
    │
    ▼
Crop → grayscale → resize 200x200 → equalise lighting
    │
    ▼
LBPH recogniser  ──► (who is it, distance score)
    │
    ▼
Right person AND distance < threshold?
    │ yes, 10 frames in a row
    ▼
Open gripper (servo on Pi / simulated on laptop) + write log entry
```

| File | What it does |
|---|---|
| `enroll.py` | Captures 40 face photos of a recipient from the webcam |
| `train.py` | Trains an LBPH face recogniser on the photos |
| `deliver.py` | Runs the delivery simulation: unlocks the gripper only for the right person |
| `gripper.py` | Servo control via `gpiozero` on a Raspberry Pi, prints a message otherwise |
| `evaluate.py` | 80/20 train/test split to report accuracy |
| `face_utils.py`, `config.py` | Shared helpers and settings |

## Setup

```bash
git clone https://github.com/<your-username>/drone-face-auth-delivery.git
cd drone-face-auth-delivery
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Enroll the recipient (and a second person to act as a stranger)
python enroll.py --name alice
python enroll.py --name bob

# 2. Train
python train.py

# 3. (optional) check accuracy
python evaluate.py

# 4. Run the delivery for alice
python deliver.py --recipient alice
```

Green box = correct recipient, orange = known but wrong person, red = unknown.
The gripper opens only after the recipient is matched in **10 consecutive frames**.
Every outcome is appended to `deliveries.csv`. Press `q` to quit.

## Design choices

- **Haar Cascade + LBPH** – both ship with OpenCV, are light enough for a Raspberry Pi,
  and are easy to explain. Deep-learning models are more accurate but heavier.
- **Histogram equalisation** – reduces the effect of dim/bright lighting.
- **Distance threshold** – LBPH returns a distance (lower = closer match). Anything above
  `CONFIDENCE_THRESHOLD` in `config.py` is treated as a stranger.
- **Consecutive-frame check** – one lucky frame can't unlock the payload.

## Limitations

- LBPH is sensitive to pose and strong lighting changes; accuracy drops with sunglasses/masks.
- A printed photo could fool it (no liveness / anti-spoofing check).
- The threshold should be tuned for your camera and lighting.

## Future work

- Swap LBPH for a deep embedding model (e.g. `face_recognition` / FaceNet)
- Add blink-based liveness detection
- Send the "unlock" command to a Pixhawk using MAVLink (`pymavlink`) after landing
- Add obstacle avoidance (ultrasonic/LiDAR) as in the full hardware design

## Tech stack

Python · OpenCV (contrib) · NumPy · gpiozero (optional, Raspberry Pi)

## Credits

Original hardware project by <teammates' names>, VIT-AP. This repository contains my own
implementation of the face-authentication software.
