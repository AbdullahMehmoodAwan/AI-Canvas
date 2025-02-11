import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import textwrap

# Initialize Streamlit components for displaying the webcam feed and AI response
st.set_page_config(layout="wide")
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
with col2:
    st.title("Answer")
    output_text_area = st.empty()

# Initialize output text
output_text = ""

# Configure the Generative AI API with your key
genai.configure(api_key="AIzaSyAhg36JztckDVW2T7Bx91lmQEbnvl7VX34")
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
# `0` refers to the default webcam. You can change this to `1`, `2`, etc., if you have multiple cameras
webcam = cv2.VideoCapture(0)
webcam.set(3, 1080)  # Set frame width
webcam.set(4, 720)  # Set frame height

# Check if the webcam is opened correctly
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the HandDetector class with the given parameters
hand_detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def detect_hand_info(frame):
    """
    Detect hands in the current frame and return hand landmarks and fingers info.
    """
    # Detect hands and return the modified frame and hand information
    hands, frame = hand_detector.findHands(frame, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        hand_data = hands[0]  # Get the first hand detected
        landmarks = hand_data["lmList"]  # List of 21 landmarks for the first hand
        fingers_up = hand_detector.fingersUp(hand_data)  # Count the number of fingers up
        return fingers_up, landmarks
    else:
        return None

def draw_on_canvas(frame, hand_info, previous_position, drawing_canvas):
    """
    Draw lines on the canvas based on the position of the index finger.
    """
    fingers_up, landmarks = hand_info
    current_position = previous_position

    # Check if the index finger is up and all other fingers are down
    if fingers_up == [0, 1, 0, 0, 0]:
        current_position = tuple(landmarks[8][0:2])  # Get the tip of the index finger
        if previous_position is not None:
            # Draw a line on the canvas from the previous position to the current position
            cv2.line(drawing_canvas, current_position, tuple(previous_position), (255, 0, 255), 10)
    elif fingers_up == [1, 0, 0, 0, 0]:
        drawing_canvas = np.zeros_like(frame)  # Clear the canvas if thumb is up
    else:
        current_position = None  # Reset current position if the finger is lifted

    return current_position, drawing_canvas

def send_to_ai(model, drawing_canvas, fingers_up):
    """
    Send the drawing to the AI model when all fingers except the thumb are up.
    """
    if fingers_up == [1, 1, 1, 1, 0]:
        # Convert the canvas to an image and send it to the AI model
        pil_image = Image.fromarray(drawing_canvas)
        response = model.generate_content(["Solve the math problem in detail", pil_image])
        return response.text
    return ""

# Initialize previous position and canvas
previous_position = None
drawing_canvas = None

# Continuously get frames from the webcam
while run:
    # Capture each frame from the webcam
    success, frame = webcam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    if drawing_canvas is None:
        # Initialize the drawing canvas as a blank image with the same dimensions as the frame
        drawing_canvas = np.zeros_like(frame)

    # Check if the frame is captured correctly
    if not success or frame is None:
        print("Error: Could not read frame from webcam.")
        break

    # Detect hand information (fingers up and landmarks)
    hand_info = detect_hand_info(frame)
    if hand_info:
        fingers_up, landmarks = hand_info
        previous_position, drawing_canvas = draw_on_canvas(frame, hand_info, previous_position, drawing_canvas)
        output_text = send_to_ai(ai_model, drawing_canvas, fingers_up)
    else:
        previous_position = None  # Reset previous position if no hand is detected

    # Combine the original frame and the canvas with drawings
    combined_frame = cv2.addWeighted(frame, 0.7, drawing_canvas, 0.3, 0)

    # Display the combined frame in the Streamlit app
    FRAME_WINDOW.image(combined_frame, channels="BGR")

    # Display the AI response in the Streamlit app
    if output_text:
        wrapped_text = textwrap.fill(output_text, width=60)
        output_text_area.text(wrapped_text)

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
