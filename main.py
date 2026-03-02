import cv2
import mediapipe as mp
import numpy as np
import time
import random
import threading
import os
import subprocess

class DoomscrollDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.roasts = [
            "Your goals > Your feed.",
            "PUT. THE. PHONE. DOWN.",
            "The algorithm wins again.",
            "Future you is disappointed.",
            "Stop scrolling. Start building."
        ]

        self.last_roast_time = 0
        self.roast_cooldown = 3
        self.current_roast = ""

        self.doomscroll_count = 0
        self.normal_count = 0
        self.detection_threshold = 3

        self.rickroll_path = "DoMixi.mp4" # change the video here, remmember to put it in to the same path :) 
        self.is_rickrolling = False
        self.rickroll_process = None

    def calculate_pitch(self, landmarks, frame_shape):
        h, w, _ = frame_shape

        nose = landmarks[1]
        chin = landmarks[152]
        forehead = landmarks[10]

        nose_point = np.array([nose.x * w, nose.y * h])
        chin_point = np.array([chin.x * w, chin.y * h])
        forehead_point = np.array([forehead.x * w, forehead.y * h])

        face_vector = chin_point - forehead_point
        vertical_vector = np.array([0, 1])

        cos_angle = np.dot(face_vector, vertical_vector) / (
            np.linalg.norm(face_vector) * np.linalg.norm(vertical_vector)
        )

        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def play_rickroll(self):
        if not self.is_rickrolling and os.path.exists(self.rickroll_path):
            self.is_rickrolling = True

            def start_video():
                if os.name == "nt":
                    os.startfile(self.rickroll_path)
                else:
                    subprocess.Popen(["xdg-open", self.rickroll_path])

            threading.Thread(target=start_video, daemon=True).start()

    def stop_rickroll(self):
        self.is_rickrolling = False

    def show_roast(self, frame):
        current_time = time.time()

        if current_time - self.last_roast_time > self.roast_cooldown:
            self.current_roast = random.choice(self.roasts)
            self.last_roast_time = current_time

        overlay = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, "DOOMSCROLLING DETECTED",
                    (w//2 - 250, 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (255, 255, 255), 2)

        cv2.putText(frame, self.current_roast,
                    (w//2 - 300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

    def run(self):
        cap = cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        print("MediaPipe Doomscroll Detector Started")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.face_mesh.process(rgb)

            raw_detection = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                pitch_angle = self.calculate_pitch(landmarks, frame.shape)

                # Debug display
                cv2.putText(frame,
                            f"Pitch: {int(pitch_angle)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                if pitch_angle > 25:  # Threshold
                    raw_detection = True

            if raw_detection:
                self.doomscroll_count += 1
                self.normal_count = 0
            else:
                self.normal_count += 1
                self.doomscroll_count = 0

            is_doomscrolling = self.doomscroll_count >= self.detection_threshold
            is_normal = self.normal_count >= self.detection_threshold

            if is_doomscrolling:
                self.show_roast(frame)
                self.play_rickroll()
            elif is_normal:
                cv2.putText(frame,
                            "Good posture!",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                self.stop_rickroll()

            cv2.imshow("Doomscrolling Blocker", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DoomscrollDetector()
    detector.run()