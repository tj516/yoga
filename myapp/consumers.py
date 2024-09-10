import cv2
import numpy as np
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from io import BytesIO
import time
from .views import models, mp, compute_similarity_score, appreciation_messages, get_average_similarity_score, pose_duration_thresholds, holis, inFrame
import mediapipe as mp
from keras.models import load_model

drawing = mp.solutions.drawing_utils
holistic = mp.solutions.pose
holis = holistic.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
pose_start_time = None
suggestions_displayed = False
appreciation_messages = [
    "Good start! Keep holding the pose.",
    "Awesome! You're doing great.",
    "Fantastic! Just a bit more.",
    "Incredible! Almost there.",
    "Excellent! You've held the pose!"
]
message_index = 0
cnt = 0
pose_duration_thresholds = [5, 10, 15, 20, 25]  # Time thresholds in seconds for each message

# Global variable to store similarity score
similarity_score = None

# List to store the recent similarity scores for averaging
similarity_scores = []
angle_data = {'shoulder': [], 'hip': [], 'knee': []}

# Global variables to store pose data
pose_data = []  # List to store pose details
pose_durations = {}  # Dictionary to store duration of each pose

# Load models
models = {
    "option_1": {"model": load_model("model.h5"), "labels": np.load("labels.npy")},
    "option_2": {"model": load_model("model_1.h5"), "labels": np.load("labels_1.npy")},
    "option_3": {"model": load_model("model_2.h5"), "labels": np.load("labels_2.npy")},
    "option_4": {"model": load_model("model_2.h5"), "labels": np.load("labels_2.npy")},
    "option_5": {"model": load_model("pu.h5"), "labels": np.load("pu.npy")},
    "option_6": {"model": load_model("test.h5"), "labels": np.load("test.npy")},
    "option_7": {"model": load_model("test1.h5"), "labels": np.load("test1.npy")},
    "option_8": {"model": load_model("Physio.h5"), "labels": np.load("Physio.npy")}
}

class VideoFeedConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.frame_count = 0
        self.pose_start_time = None
        self.suggestions_displayed = False
        self.message_index = 0
        self.cnt = 0
        self.similarity_score = None
        self.similarity_scores = []
        self.pose_data = []
        self.pose_durations = {}
        self.reference_pose = None  # Placeholder for the reference pose

    async def disconnect(self, close_code):
        await self.close()

    async def receive(self, bytes_data=None):
        if bytes_data:
            # Decode the video frame
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frm = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frm is not None:
                self.cnt += 1
                print("Frame no.", self.cnt)
                
                # Reduce resolution to reduce data size (e.g., 320x180 for 480p)
                desired_width = 640  # Reduced from 640
                aspect_ratio = 16 / 9  # Standard 16:9 aspect ratio
                desired_height = int(desired_width / aspect_ratio)
                frm = cv2.resize(frm, (desired_width, desired_height))
                
                # Create a blank window that matches the target size (16:9 ratio)
                window = np.zeros((desired_height, desired_width, 3), dtype="uint8")
                
                # Flip the frame horizontally (mirror effect)
                frm = cv2.flip(frm, 1)
                
                # Process the frame (similar to generate_frames1)
                res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                frm = cv2.blur(frm, (4, 4))

                self.similarity_score = None  # Initialize similarity_score variable
                
                if res.pose_landmarks:
                    lst = []  # Initialize the list before appending

                    if self.pose_start_time is None:
                        self.pose_start_time = time.time()
                        self.message_index = 0

                    in_frame, suggestions = inFrame(res.pose_landmarks.landmark)
                    if in_frame:
                        if suggestions:
                            pose_start_time = time.time()
                            suggestions_displayed = False
                            message_index = 0

                        for suggestion_index, suggestion in enumerate(suggestions):
                            cv2.putText(frm, suggestion, (20, 50 + 30 * suggestion_index), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        for i in res.pose_landmarks.landmark:
                            lst.append(i.x - res.pose_landmarks.landmark[0].x)
                            lst.append(i.y - res.pose_landmarks.landmark[0].y)

                        lst = np.array(lst).reshape(1, -1)

                        model_data = models['option_8']
                        model = model_data["model"]
                        labels = model_data["labels"]

                        p = model.predict(lst)
                        pred = labels[np.argmax(p)]

                        if p[0][np.argmax(p)] > 0.75:
                            cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)

                            # Compute similarity score
                            if self.reference_pose is None:
                                self.reference_pose = res.pose_landmarks.landmark
                            similarity_score = compute_similarity_score(res.pose_landmarks.landmark, self.reference_pose)
                            
                            # Add the similarity score to the list
                            self.similarity_scores.append(similarity_score)
                            if len(self.similarity_scores) > 30:  # Limit to last 30 frames
                                self.similarity_scores.pop(0)

                            avg_similarity_score = get_average_similarity_score()
                            # Ensure avg_similarity_score is a float before formatting
                            try:
                                avg_similarity_score = float(avg_similarity_score)
                            except ValueError:
                                avg_similarity_score = 0.0  # Handle the case where conversion fails
                            cv2.putText(frm, f"Similarity: {avg_similarity_score:.1f}/10", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            elapsed_time = time.time() - self.pose_start_time
                            pose_name = pred
                            if pose_name not in self.pose_durations:
                                self.pose_durations[pose_name] = 0
                            self.pose_durations[pose_name] += elapsed_time

                            pose_feedback = appreciation_messages[self.message_index]
                            self.pose_data.append({"name": pose_name, "duration": elapsed_time, "feedback": pose_feedback})

                            if elapsed_time >= pose_duration_thresholds[self.message_index]:
                                cv2.putText(frm, appreciation_messages[self.message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                if self.message_index < len(appreciation_messages) - 1:
                                    self.message_index += 1
                            else:
                                cv2.putText(frm, appreciation_messages[self.message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

                else:
                    self.pose_start_time = None
                    self.suggestions_displayed = False
                    self.message_index = 0
                    cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                    connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

                # Place the processed frame in the window
                window[0:desired_height, 0:desired_width, :] = frm

                # Convert frame to bytes with reduced quality and send it as a WebSocket message
                ret, buffer = cv2.imencode('.jpg', window, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Reduced quality to 70%
                frame = buffer.tobytes()
                
                # Send similarity score as a separate WebSocket message
                if self.similarity_score is not None:
                    similarity_score_str = f"Similarity: {self.similarity_score:.1f}/10"
                    await self.send(text_data=similarity_score_str)
                    
                await self.send(None, frame)
            else:
                await asyncio.sleep(1)