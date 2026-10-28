"""Payload gripper. Uses a real servo on a Raspberry Pi, otherwise just simulates."""
import config

try:
    from gpiozero import AngularServo
    _HAS_GPIO = True
except Exception:          # library missing, or not running on a Pi
    _HAS_GPIO = False


class Gripper:
    def __init__(self):
        self.servo = AngularServo(config.SERVO_PIN, min_angle=0, max_angle=90) if _HAS_GPIO else None
        self.is_open = False
        self.close()

    def open(self):
        """Release the payload."""
        if self.servo:
            self.servo.angle = 90
        self.is_open = True
        print("[GRIPPER] OPEN  -> payload released")

    def close(self):
        """Hold the payload."""
        if self.servo:
            self.servo.angle = 0
        self.is_open = False
        print("[GRIPPER] CLOSED -> payload secured")
