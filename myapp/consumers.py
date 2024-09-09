import cv2
import numpy as np
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from io import BytesIO
import time
from .views import models, mp, compute_similarity_score, appreciation_messages, get_average_similarity_score, pose_duration_thresholds, holis, inFrame

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
                desired_width = 320  # Reduced from 640
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
                    if self.pose_start_time is None:
                        self.pose_start_time = time.time()
                        self.message_index = 0

                    in_frame, suggestions = inFrame(res.pose_landmarks.landmark)
                    if in_frame:
                        # Display suggestions
                        for suggestion_index, suggestion in enumerate(suggestions):
                            cv2.putText(frm, suggestion, (20, 50 + 30 * suggestion_index),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # Predict pose
                        lst = [coord for lm in res.pose_landmarks.landmark for coord in (lm.x, lm.y)]
                        lst = np.array(lst).reshape(1, -1)
                        model_data = models['option_1']
                        model = model_data["model"]
                        labels = model_data["labels"]

                        p = model.predict(lst)
                        pred = labels[np.argmax(p)]

                        if p[0][np.argmax(p)] > 0.75:
                            cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
                            
                            if self.reference_pose is None:
                                self.reference_pose = res.pose_landmarks.landmark
                            
                            # Compute similarity score
                            self.similarity_score = compute_similarity_score(res.pose_landmarks.landmark, self.reference_pose)
                            self.similarity_scores.append(self.similarity_score)
                            if len(self.similarity_scores) > 30:
                                self.similarity_scores.pop(0)

                            avg_similarity_score = get_average_similarity_score(self.similarity_scores)
                            cv2.putText(frm, f"Similarity: {avg_similarity_score:.1f}/10",
                                        (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            elapsed_time = time.time() - self.pose_start_time
                            pose_name = pred
                            self.pose_durations[pose_name] = self.pose_durations.get(pose_name, 0) + elapsed_time

                            pose_feedback = appreciation_messages[self.message_index]
                            self.pose_data.append({"name": pose_name, "duration": elapsed_time, "feedback": pose_feedback})

                            if elapsed_time >= pose_duration_thresholds[self.message_index]:
                                cv2.putText(frm, appreciation_messages[self.message_index],
                                            (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                if self.message_index < len(appreciation_messages) - 1:
                                    self.message_index += 1
                            else:
                                cv2.putText(frm, appreciation_messages[self.message_index],
                                            (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frm, "Asana is either wrong or not trained",
                                        (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)
                else:
                    self.pose_start_time = None
                    self.suggestions_displayed = False
                    self.message_index = 0
                    cv2.putText(frm, "Make Sure Full body visible",
                                (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                # Draw pose landmarks
                drawing = mp.solutions.drawing_utils
                holistic = mp.solutions.pose
                drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                       connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                       landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

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
