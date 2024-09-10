from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators import gzip
import cv2
import time
import math
import numpy as np
import mediapipe as mp
from keras.models import load_model
import logging
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle,
    Spacer, Image, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.barcode import qr
from io import BytesIO
from django.http import HttpResponse
from matplotlib import pyplot as plt
#from myapp.video_capture import WebSocketVideoCapture
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPM
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageDraw

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

logger = logging.getLogger(__name__)
#cap = WebSocketVideoCapture()

# Global variable to store similarity score
similarity_score = None

# List to store the recent similarity scores for averaging
similarity_scores = []
angle_data = {'shoulder': [], 'hip': [], 'knee': []}

# Global variables to store pose data
pose_data = []  # List to store pose details
pose_durations = {}  # Dictionary to store duration of each pose

# Custom Header/Footer functions
def header_footer(canvas, doc):
    canvas.saveState()

    # Header
    header_text = "Yoga Pose Session Report"
    canvas.setFont('Helvetica-Bold', 12)
    canvas.drawString(inch, A4[1] - 0.75 * inch, header_text)

    # Footer
    footer_text = f"Page {doc.page}"
    canvas.setFont('Helvetica', 10)
    canvas.drawString(inch, 0.75 * inch, footer_text)

    canvas.restoreState()

def angle_between_points(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = math.degrees(radians)
    angle = abs(angle) if angle <= 180 else 360 - abs(angle)
    return angle

def inFrame(lst):
    shoulder_angle_threshold = 20
    hip_angle_threshold = 20
    elbow_angle_threshold = 20
    knee_angle_threshold = 20

    if lst[27].visibility > 0.6 or lst[28].visibility > 0.6 and \
       lst[15].visibility > 0.6 and lst[16].visibility > 0.6:

        shoulder_angle = angle_between_points((lst[23].x, lst[23].y),
                                              (lst[25].x, lst[25].y),
                                              (lst[27].x, lst[27].y))
        elbow_angle_left = angle_between_points((lst[11].x, lst[11].y),
                                                (lst[13].x, lst[13].y),
                                                (lst[15].x, lst[15].y))
        elbow_angle_right = angle_between_points((lst[12].x, lst[12].y),
                                                 (lst[14].x, lst[14].y),
                                                 (lst[16].x, lst[16].y))
        hip_angle = angle_between_points((lst[11].x, lst[11].y),
                                          (lst[13].x, lst[13].y),
                                          (lst[15].x, lst[15].y))
        knee_angle_left = angle_between_points((lst[13].x, lst[13].y),
                                               (lst[15].x, lst[15].y),
                                               (lst[27].x, lst[27].y))
        knee_angle_right = angle_between_points((lst[14].x, lst[14].y),
                                                (lst[16].x, lst[16].y),
                                                (lst[28].x, lst[28].y))

        suggestions = []
        if shoulder_angle < shoulder_angle_threshold:
            suggestions.append("Improvement: Shoulders should be wider.")
        if elbow_angle_left < elbow_angle_threshold or elbow_angle_right < elbow_angle_threshold:
            suggestions.append("Improvement: Elbows should be more straight.")
        if hip_angle < hip_angle_threshold:
            suggestions.append("Improvement: Hips should be wider.")
        if knee_angle_left < knee_angle_threshold or knee_angle_right < knee_angle_threshold:
            suggestions.append("Improvement: Knees should be more straight.")

        # Update angle_data with new values
        angle_data['shoulder'].append(shoulder_angle)
        angle_data['hip'].append(hip_angle)
        angle_data['knee'].append((knee_angle_left + knee_angle_right) / 2)

        return True, suggestions
    else:
        return False, []

def count_repetitions(angles, criteria):
    global repetition_count, down_position
    """
    Count repetitions based on the angles of body parts specified by the exercise criteria.
    Args:
    - angles: Dictionary of angles calculated for various joints.
    - criteria: Dictionary containing the angle threshold and body part for the current exercise.
    
    Returns:
    - repetition_count: Updated count of repetitions.
    """
    if criteria['body_part'] not in angles:
        return repetition_count  # If the required body part is not in the angles, return current count.

    angle = angles[criteria['body_part']]

    if not down_position and isinstance(angle, (int, float)) and angle < criteria['angle_threshold']:
        down_position = True  # Body is in the down position.
    elif down_position and isinstance(angle, (int, float)) and angle > criteria['angle_threshold']:
        repetition_count += 1  # Increment repetitions when moving from down to up.
        down_position = False  # Reset down position for the next repetition.

    return repetition_count

def get_angles(pose_landmarks):
    """
    Calculate the angles for key joints (shoulder, hip, knee) based on the pose landmarks.
    
    Args:
    - pose_landmarks: Pose landmarks obtained from MediaPipe Holistic model.
    
    Returns:
    - Dictionary with calculated angles for each joint.
    """
    angles = {}
    
    if pose_landmarks:
        # Example calculation for the shoulder, hip, and knee angles.
        angles['shoulder'] = angle_between_points(
            (pose_landmarks[11].x, pose_landmarks[11].y),  # Left shoulder
            (pose_landmarks[13].x, pose_landmarks[13].y),  # Left elbow
            (pose_landmarks[15].x, pose_landmarks[15].y)   # Left wrist
        )
        angles['hip'] = angle_between_points(
            (pose_landmarks[23].x, pose_landmarks[23].y),  # Left hip
            (pose_landmarks[25].x, pose_landmarks[25].y),  # Left knee
            (pose_landmarks[27].x, pose_landmarks[27].y)   # Left ankle
        )
        angles['knee'] = angle_between_points(
            (pose_landmarks[25].x, pose_landmarks[25].y),  # Left knee
            (pose_landmarks[27].x, pose_landmarks[27].y),  # Left ankle
            (pose_landmarks[29].x, pose_landmarks[29].y)   # Left foot
        )
    
    return angles


def compute_similarity_score(current_pose, reference_pose):
    global similarity_score  # Declare global variable
    # Calculate Euclidean distance between current_pose and reference_pose landmarks
    distances = [math.sqrt((c.x - r.x)**2 + (c.y - r.y)**2) for c, r in zip(current_pose, reference_pose)]
    # Normalize the distances and compute similarity score between 1 and 10
    mean_distance = np.mean(distances)
    similarity_score = max(1, min(10, 10 * (1 - mean_distance)))  # Scale score to 1-10
    return similarity_score

def get_average_similarity_score():
    global similarity_scores
    if not similarity_scores:
        return "N/A"
    return sum(similarity_scores) / len(similarity_scores)

def get_average_angles():
    avg_angles = {}
    for key, values in angle_data.items():
        if values:
            avg_angles[key] = sum(values) / len(values)
        else:
            avg_angles[key] = "N/A"
    return avg_angles

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

holistic = mp.solutions.pose
holis = holistic.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        pose_option = request.POST.get('pose_option', 'option_1')
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        uploaded_file_url = fs.url(filename)
        return JsonResponse({'video_url': uploaded_file_url})
    return JsonResponse({'error': 'Failed to upload video'}, status=400)

def process_uploaded_video(video_path, pose_option):
    global pose_start_time, suggestions_displayed, message_index, similarity_score, similarity_scores
    global pose_data, pose_durations  # Add global variables

    cap_video = cv2.VideoCapture(video_path)
    if not cap_video.isOpened():
        logger.error("Failed to open video file: %s", video_path)
        return None
    
    output_path = video_path.replace('.mp4', '_processed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    reference_pose = None  # Placeholder for the reference pose

    while True:
        lst = []
        ret, frm = cap_video.read()
        if not ret:
            break

        window = np.zeros((940, 940, 3), dtype="uint8")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        similarity_score = None  # Initialize similarity_score variable

        if res.pose_landmarks:
            if pose_start_time is None:
                pose_start_time = time.time()
                message_index = 0

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

                model_data = models[pose_option]
                model = model_data["model"]
                labels = model_data["labels"]

                p = model.predict(lst)
                pred = labels[np.argmax(p)]

                if p[0][np.argmax(p)] > 0.75:
                    cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)

                    # Compute similarity score
                    if reference_pose is None:
                        reference_pose = res.pose_landmarks.landmark
                    similarity_score = compute_similarity_score(res.pose_landmarks.landmark, reference_pose)
                    
                    # Add the similarity score to the list
                    similarity_scores.append(similarity_score)
                    if len(similarity_scores) > 30:  # Limit to last 30 frames
                        similarity_scores.pop(0)

                    avg_similarity_score = get_average_similarity_score()
                    cv2.putText(frm, f"Similarity: {avg_similarity_score:.1f}/10", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    elapsed_time = time.time() - pose_start_time
                    pose_name = pred
                    if pose_name not in pose_durations:
                        pose_durations[pose_name] = 0
                    pose_durations[pose_name] += elapsed_time

                    pose_feedback = appreciation_messages[message_index]
                    pose_data.append({"name": pose_name, "duration": elapsed_time, "feedback": pose_feedback})

                    if elapsed_time >= pose_duration_thresholds[message_index]:
                        cv2.putText(frm, appreciation_messages[message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if message_index < len(appreciation_messages) - 1:
                            message_index += 1
                    else:
                        cv2.putText(frm, appreciation_messages[message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)
        
        else:
            pose_start_time = None
            suggestions_displayed = False
            message_index = 0
            cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                               connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

        window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))

        ret, buffer = cv2.imencode('.jpg', window)
        frame = buffer.tobytes()

        # Append similarity score to the frame data
        if similarity_score is not None:
            frame += b"\n" + f"Similarity: {similarity_score:.1f}/10".encode()

        out.write(cv2.imdecode(np.frombuffer(frame, np.uint8), -1))
    
    cap_video.release()
    out.release()
    return output_path

def generate_frames(request, cap, pose_option='option_1'):
    global pose_start_time, suggestions_displayed, message_index, cnt, similarity_score, similarity_scores
    global pose_data, pose_durations  # Add global variables

    reference_pose = None
    
    # Adjust frame rate by skipping frames
    frame_skip_rate = 2  # Capture every second frame for 15 FPS if original is 30 FPS

    while True:
        print("frame no ", cnt)
        cnt += 1
	# Capture only every nth frame
        if cnt % frame_skip_rate != 0:
            continue
        lst = []
        ret, frm = cap.read()
        if not ret:
            logger.error("Failed to capture frame from webcam")
            break

	# Resize frame to match 16:9 aspect ratio (e.g., 640x360 for 480p)
        desired_width = 640
        aspect_ratio = 16 / 9  # Standard 16:9 aspect ratio
        desired_height = int(desired_width / aspect_ratio)
        frm = cv2.resize(frm, (desired_width, desired_height))

        # Create a blank window that matches the target size (16:9 ratio)
        window = np.zeros((desired_height, desired_width, 3), dtype="uint8")

        # Flip the frame horizontally (mirror effect)
        frm = cv2.flip(frm, 1)

        # Convert the frame to RGB and process with holistic model
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # Blur the frame for a smoother look
        frm = cv2.blur(frm, (4, 4))

        similarity_score = None  # Initialize similarity_score variable

        if res.pose_landmarks:
            if pose_start_time is None:
                pose_start_time = time.time()
                message_index = 0

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

                model_data = models[pose_option]
                model = model_data["model"]
                labels = model_data["labels"]

                p = model.predict(lst)
                pred = labels[np.argmax(p)]

                if p[0][np.argmax(p)] > 0.75:
                    cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)

                    # Compute similarity score
                    if reference_pose is None:
                        reference_pose = res.pose_landmarks.landmark
                    similarity_score = compute_similarity_score(res.pose_landmarks.landmark, reference_pose)
                    
                    # Add the similarity score to the list
                    similarity_scores.append(similarity_score)
                    if len(similarity_scores) > 30:  # Limit to last 30 frames
                        similarity_scores.pop(0)

                    avg_similarity_score = get_average_similarity_score()
                    cv2.putText(frm, f"Similarity: {avg_similarity_score:.1f}/10", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		    # Compute angles for joints and count repetitions
                    angles = get_angles(res.pose_landmarks.landmark)
                    repetition_count = count_repetitions(angles, exercise_criteria)

                    # Display the repetition count on the top-left corner of the frame
                    # Adjusted coordinates to (10, 50) for top-left placement
                    cv2.putText(frm, f"Reps: {repetition_count}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)


                    elapsed_time = time.time() - pose_start_time
                    pose_name = pred
                    if pose_name not in pose_durations:
                        pose_durations[pose_name] = 0
                    pose_durations[pose_name] += elapsed_time

                    pose_feedback = appreciation_messages[message_index]
                    pose_data.append({"name": pose_name, "duration": elapsed_time, "feedback": pose_feedback})

                    if elapsed_time >= pose_duration_thresholds[message_index]:
                        cv2.putText(frm, appreciation_messages[message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if message_index < len(appreciation_messages) - 1:
                            message_index += 1
                    else:
                        cv2.putText(frm, appreciation_messages[message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

        else:
            pose_start_time = None
            suggestions_displayed = False
            message_index = 0
            cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                               connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

        # Place the processed frame in the window
        window[0:desired_height, 0:desired_width, :] = frm

        ret, buffer = cv2.imencode('.jpg', window)
        frame = buffer.tobytes()

        # Append similarity score to the frame data
        if similarity_score is not None:
            frame += b"\n" + f"Similarity: {similarity_score:.1f}/10".encode()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
	       + f'Reps: {repetition_count}'.encode() + b'\r\n')

#def start_socket(request):
#    return JsonResponse({'success': 'started'}, status=200)

def generate_frames1(pose_option='option_1'):
    global pose_start_time, suggestions_displayed, message_index, cnt, similarity_score, similarity_scores
    global pose_data, pose_durations  # Add global variables
    
    reference_pose = None

    # Adjust frame rate by skipping frames
    frame_skip_rate = 2  # Capture every second frame for 15 FPS if original is 30 FPS
    
    while True:
        if not cap.queue_empty():
            print("frame no. ", cnt)
            cnt += 1
	    # Capture only every nth frame
            if cnt % frame_skip_rate != 0:
            	continue
            lst = []
            ret, frm = cap.read()
            if not ret:
                logger.error("Failed to capture frame from webcam mayank")
                time.sleep(1)
                continue

	    # Resize frame to match 16:9 aspect ratio (e.g., 640x360 for 480p)
            desired_width = 640
            aspect_ratio = 16 / 9  # Standard 16:9 aspect ratio
            desired_height = int(desired_width / aspect_ratio)
            frm = cv2.resize(frm, (desired_width, desired_height))

            # Create a blank window that matches the target size (16:9 ratio)
            window = np.zeros((desired_height, desired_width, 3), dtype="uint8")

            # Flip the frame horizontally (mirror effect)
            frm = cv2.flip(frm, 1)

            # Convert the frame to RGB and process with holistic model
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            # Blur the frame for a smoother look
            frm = cv2.blur(frm, (4, 4))

            similarity_score = None  # Initialize similarity_score variable

            if res.pose_landmarks:
                if pose_start_time is None:
                    pose_start_time = time.time()
                    message_index = 0

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

                    model_data = models[pose_option]
                    model = model_data["model"]
                    labels = model_data["labels"]

                    p = model.predict(lst)
                    pred = labels[np.argmax(p)]

                    if p[0][np.argmax(p)] > 0.75:
                        cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)

                        # Compute similarity score
                        if reference_pose is None:
                            reference_pose = res.pose_landmarks.landmark
                        similarity_score = compute_similarity_score(res.pose_landmarks.landmark, reference_pose)
                        
                        # Add the similarity score to the list
                        similarity_scores.append(similarity_score)
                        if len(similarity_scores) > 30:  # Limit to last 30 frames
                            similarity_scores.pop(0)

                        avg_similarity_score = get_average_similarity_score()
                        cv2.putText(frm, f"Similarity: {avg_similarity_score:.1f}/10", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

			# Compute angles for joints and count repetitions
                        angles = get_angles(res.pose_landmarks.landmark)
                        repetition_count = count_repetitions(angles, exercise_criteria)

                    	# Display the repetition count on the top-left corner of the frame
                    	# Adjusted coordinates to (10, 50) for top-left placement
                        cv2.putText(frm, f"Reps: {repetition_count}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)


                        elapsed_time = time.time() - pose_start_time
                        pose_name = pred
                        if pose_name not in pose_durations:
                            pose_durations[pose_name] = 0
                        pose_durations[pose_name] += elapsed_time

                        pose_feedback = appreciation_messages[message_index]
                        pose_data.append({"name": pose_name, "duration": elapsed_time, "feedback": pose_feedback})

                        if elapsed_time >= pose_duration_thresholds[message_index]:
                            cv2.putText(frm, appreciation_messages[message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            if message_index < len(appreciation_messages) - 1:
                                message_index += 1
                        else:
                            cv2.putText(frm, appreciation_messages[message_index], (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)
            
            else:
                pose_start_time = None
                suggestions_displayed = False
                message_index = 0
                cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

            # Place the processed frame in the window
            window[0:desired_height, 0:desired_width, :] = frm

            ret, buffer = cv2.imencode('.jpg', window)
            frame = buffer.tobytes()

            # Append similarity score to the frame data and log it
            if similarity_score is not None:
                similarity_score_str = f"Similarity: {similarity_score:.1f}/10"
                frame += b"\n" + similarity_score_str.encode()
                logger.debug(f"Appending similarity score: {similarity_score_str}")

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
		+ f'Reps: {repetition_count}'.encode() + b'\r\n')
        else:
            time.sleep(1)

@gzip.gzip_page
@csrf_exempt
def video_feed(request):
    capture_type = request.GET.get('capture_type', 'live')
    pose_option = request.GET.get('pose_option', 'option_1')
    global exercise_option

    exercise_option = request.GET.get('exercise_option', 'pushup')  # Set exercise option based on the request

    if capture_type == 'recorded_video' and request.method == "POST" and request.FILES.get('video'):
        try:
            video = request.FILES['video']
            title = str(datetime.datetime.now())
            video_name = title + '_' + video.name
            video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_name)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            with open(video_path, 'wb+') as destination:
                for chunk in video.chunks():
                    destination.write(chunk)

            return JsonResponse({'message': 'Video uploaded successfully', 'file_path': video_path}, status=201)
        except:
            return JsonResponse({'message': 'unable to save video'}, status=500)
    elif capture_type == 'live':
        try:
            return StreamingHttpResponse(generate_frames1(pose_option=pose_option), content_type="multipart/x-mixed-replace;boundary=frame")
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    elif capture_type == 'video':
        video_path = request.GET.get('video_path', '')
        if video_path:
            video_path = os.path.join(settings.MEDIA_ROOT, video_path)
            cap_video = cv2.VideoCapture(video_path)
            if not cap_video.isOpened():
                logger.error("Failed to open video file: %s", video_path)
                return JsonResponse({'error': 'Failed to open video file'}, status=500)
            return StreamingHttpResponse(generate_frames(request, cap_video, pose_option), content_type="multipart/x-mixed-replace;boundary=frame")

    return render(request, 'yoga_dashboard.html')

# View for rendering the yoga.html template
def yoga_home(request):
    return render(request, 'yoga.html')

def home(request):
    return render(request, 'yoga.html')

def follow(request):
    return render(request, 'yoga.html')

# View for rendering the yoga_dashboard.html template
def yoga_dashboard(request):
    return render(request, 'yoga_dashboard.html')

def yoga_file(request):
    return render(request, 'yoga_file.html')

def get_similarity_score1(request):
    # Extract the pose option or other parameters if needed
    pose_option = request.GET.get('pose_option', 'option_1')

    # Calculate the similarity score here
    similarity_score1 = get_average_similarity_score()  # Replace with actual calculation

    # Convert the similarity score from 0-10 to 0-100
    similarity_score = similarity_score1 * 10

    # Ensure the score is within 0-100 range
    similarity_score = max(0, min(100, similarity_score))

    # Determine the feedback message based on the similarity score
    if similarity_score < 50:
        feedback_message = "Try to correct the pose"
    elif 50 <= similarity_score < 70:
        feedback_message = "Good, keep it up"
    elif 70 <= similarity_score < 90:
        feedback_message = "You are doing great"
    else:
        feedback_message = "Superb, you have done the pose correctly"

    # Return the similarity score as JSON
    return JsonResponse({'similarity_score': similarity_score, 'feedback_message': feedback_message})


# View to return similarity score
@csrf_exempt
def get_similarity_score(request):
    avg_similarity_score = get_average_similarity_score()
    return JsonResponse({'similarity_score': avg_similarity_score})

@csrf_exempt
def generate_report_summary(pose_data, avg_similarity_score, avg_angles, total_time):
    summary = []
    summary.append(f"Average Similarity Score: {avg_similarity_score:.2f}/10" if isinstance(avg_similarity_score, float) else avg_similarity_score)
    summary.append(f"Average Shoulder Angle: {avg_angles['shoulder']:.2f} degrees" if isinstance(avg_angles['shoulder'], float) else avg_angles['shoulder'])
    summary.append(f"Average Hip Angle: {avg_angles['hip']:.2f} degrees" if isinstance(avg_angles['hip'], float) else avg_angles['hip'])
    summary.append(f"Average Knee Angle: {avg_angles['knee']:.2f} degrees" if isinstance(avg_angles['knee'], float) else avg_angles['knee'])
    summary.append(f"Total Session Time: {total_time} seconds")

    pose_durations_summary = "\n".join([f"{pose['name']}: {pose['duration']:.2f} s - {pose['feedback']}" for pose in pose_data])

    summary.append("Pose Durations:")
    summary.append(pose_durations_summary)
    
    return "\n".join(summary)

@csrf_exempt
def generate_report(request):
    avg_similarity_score = get_average_similarity_score()
    avg_angles = get_average_angles()
    total_time = sum(pose_duration_thresholds)

    dynamic_pose_data = []
    for pose_name, duration in pose_durations.items():
        feedback = "Keep it up!"
        dynamic_pose_data.append({"name": pose_name, "duration": duration, "feedback": feedback})

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'

    doc = BaseDocTemplate(response, pagesize=A4)
    frame1 = Frame(doc.leftMargin, doc.bottomMargin + 0.5 * inch, doc.width, doc.height - 1.5 * inch, id='frame1')

    def add_background(canvas, doc):
        # bg_image_path = '/myapp/bg.jpg'
        bg_image_path = os.path.join(settings.BASE_DIR, 'myapp', 'bg.jpg')
        bg_image = PILImage.open(bg_image_path)
        canvas.drawImage(bg_image_path, 0, 0, width=A4[0], height=A4[1], mask='auto')

    template1 = PageTemplate(id='FirstPage', frames=[frame1], onPage=add_background)
    doc.addPageTemplates([template1])

    elements = []

    # Add company logo
    #logo_path = '/root/projectdi/myapp/Logo.png'  # Replace with the actual path to the logo
    #if os.path.exists(logo_path):
    #    try:
    #        logo = Image(logo_path)
    #        # Adjust the size proportionally
    #        logo.drawHeight = 0.05 * inch
    #        logo.drawWidth = 2.5 * inch * (logo.imageWidth / logo.imageHeight)  # Maintain aspect ratio
    #        logo.hAlign = 'CENTER'
    #        elements.append(logo)
    #    except Exception as e:
    #        logger.error(f"Error adding logo to PDF: {e}")
    #else:
    #    logger.error(f"Logo file not found at: {logo_path}")

    # Add the title with styling
    styles = getSampleStyleSheet()
    title = Paragraph("<b><font size=18 color='#4A90E2'>Yoga Pose Session Report</font></b>", styles['Title'])
    elements.append(Spacer(1, 12))
    elements.append(title)
    elements.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Average Similarity Score", f"{avg_similarity_score:.2f}/10" if isinstance(avg_similarity_score, float) else avg_similarity_score],
        ["Total Session Time", f"{total_time} seconds"]
    ]

    table = Table(data, colWidths=[200, 50, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4A90E2")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E5F7FF")),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b><font size=14 color='#4A90E2'>Pose Analysis with Gradient Bars</font></b>", styles['Title']))
    elements.append(Spacer(1, 12))

    pose_data = [["Pose Name", "Duration (s)", "Feedback", "Angle Accuracy"]]
    for pose in dynamic_pose_data:
        angle_accuracy_image = generate_gradient_bar_image(pose['name'], avg_angles, avg_similarity_score)
        pose_data.append([pose["name"], f"{pose['duration']} s", pose["feedback"], angle_accuracy_image])

    pose_table = Table(pose_data, colWidths=[150, 100, 200, 150])
    pose_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4A90E2")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E5F7FF")),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(pose_table)
    elements.append(Spacer(1, 12))

    report_summary = generate_report_summary(dynamic_pose_data, avg_similarity_score, avg_angles, total_time)
    summary_title = Paragraph("<b><font size=14 color='#4A90E2'>Session Summary</font></b>", styles['Title'])
    elements.append(summary_title)
    elements.append(Spacer(1, 12))
    summary_paragraph = Paragraph(report_summary, styles['BodyText'])
    elements.append(summary_paragraph)
    elements.append(Spacer(1, 12))

    similarity_scores_over_time = [similarity_score for similarity_score in similarity_scores]
    plt.figure(figsize=(6, 4))
    plt.plot(similarity_scores_over_time, marker='o', color='#4A90E2')
    plt.title("Similarity Score Over Time")
    plt.xlabel("Time (frames)")
    plt.ylabel("Similarity Score")
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format="PNG")
    buffer.seek(0)
    img = Image(buffer)
    img.drawHeight = 3 * inch
    img.drawWidth = 6 * inch
    elements.append(img)

    feedback_title = Paragraph("<b><font size=14 color='#4A90E2'>Feedback and Tips</font></b>", styles['Title'])
    elements.append(feedback_title)
    elements.append(Spacer(1, 12))

    feedback_text = """
        <font color='#4A90E2'>Keep up the great work!</font> Remember to maintain proper alignment in your poses and breathe deeply. 
        Focus on holding each pose steadily for better results. Consider working on improving balance in your poses for better symmetry.
    """
    feedback_paragraph = Paragraph(feedback_text, styles['BodyText'])
    elements.append(feedback_paragraph)

    doc.build(elements)
    return response

def generate_gradient_bar_image(pose_name, avg_angles, avg_similarity_score):
    bar_width = 100
    bar_height = 20
    gradient_steps = 100

    # Create a new image with a white background
    gradient_image = PILImage.new('RGB', (bar_width, bar_height), "white")
    draw = ImageDraw.Draw(gradient_image)

    # Draw the gradient
    for i in range(gradient_steps):
        color_value = i / gradient_steps
        r = int(255 * (1 - color_value))
        g = int(255 * color_value)
        b = 0
        color = (r, g, b)
        step_width = bar_width / gradient_steps
        draw.rectangle([i * step_width, 0, (i + 1) * step_width, bar_height], fill=color)

    # Add a marker for the score
    score_position = calculate_marker_position(avg_similarity_score)
    draw.rectangle([score_position - 2, 0, score_position + 2, bar_height], fill=(0, 0, 0))

    # Save to buffer
    buffer = BytesIO()
    gradient_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Convert to ReportLab Image
    return Image(buffer)

def calculate_marker_position(score, max_score=10):
    bar_width = 100
    position = (score / max_score) * bar_width
    return position