import math
import cv2
import mediapipe as mp
import pyautogui as ag
ag.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

leftMargin = 0.1
rightMargin = 0.1
topMargin = 0.1
bottomMargin = 0.4

def distance(a, b) -> float:
  _x = a.x - b.x
  _y = a.y - b.y
  return math.sqrt(_x*_x + _y*_y)

def detectGesture(landmarks, unit = float) -> str:
  if distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
              landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) < 0.225*unit:
    return "leftclick"
  else:
    if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
          landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
          landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and \
          landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
      return "point"
    else:
      return "none"

    # For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
  hand_unit = 0.0
  while cap.isOpened():
    gesture = "none"
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      mp_drawing.draw_landmarks(
         image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      if distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])< 1.75 * distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                                                                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]) and distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                                                                                                                                                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])> 1.25 * distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]):
          hand_unit = distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                 hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
          gesture = detectGesture(hand_landmarks, hand_unit)
      if gesture == "point" and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > leftMargin and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < 1-rightMargin \
              and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > topMargin and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < 1-bottomMargin:
        ag.moveTo((results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x-leftMargin)*(1/(1-leftMargin-rightMargin))*screensize[0], (results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y-topMargin)*(1/(1-topMargin-bottomMargin))*screensize[1], 0.1)
      if gesture == "leftclick":
        ag.click()

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 100)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(image, gesture + " " + str(hand_unit),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()