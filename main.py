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

leftMargin = 0.3
rightMargin = 0.3
topMargin = 0.2
bottomMargin = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 100)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

LEFT_CLICK = 1
RIGHT_CLICK = 2
POINTER = 4
SCROLL = 8

def distance(a, b) -> float:
  _x = a.x - b.x
  _y = a.y - b.y
  return math.sqrt(_x*_x + _y*_y)

def detectGesture(landmarks, unit = float) -> int:
  result = 0
  if distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
              landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]) < 0.21*unit and landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
     distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                   landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) > 0.4 * unit:
    result = result | RIGHT_CLICK

  if distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
              landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) < 0.25*unit and landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
     distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                   landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) > 0.4 * unit:
    result = result | LEFT_CLICK

    # landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
  if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.PINKY_MCP].y or \
     distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                   landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.3 * unit:
    result = result | POINTER

  if distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                   landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.3 * unit:
    result = result | SCROLL
  return result

    # For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
  hand_unit = 0.0
  previousX = 0
  previousY = 0
  leftClicked = False
  rightClicked = False
  while cap.isOpened():
    gesture = 0
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

    #If a hand is detected - process the gestures
    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      #Calculate hand measurement unit if the hand is facing the camera
      if distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])< 1.75 * distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                                                                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]) and distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                                                                                                                                                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])> 1.25 * distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]):
          hand_unit = distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
          gesture = detectGesture(hand_landmarks, hand_unit)

      #Check for left click
      if (gesture & LEFT_CLICK) == LEFT_CLICK and leftClicked == False:
        ag.mouseDown(button='left')
        leftClicked = True
      else:
        if (gesture & LEFT_CLICK) != LEFT_CLICK and leftClicked == True:
            ag.mouseUp(button='left')
            leftClicked = False

      #Check for right click
      if (gesture & RIGHT_CLICK) == RIGHT_CLICK and rightClicked == False:
        ag.mouseDown(button='right')
        rightClicked = True
      else:
        if (gesture & RIGHT_CLICK) != RIGHT_CLICK and rightClicked == True:
           ag.mouseUp(button='right')
           rightClicked = False

      #Check for pointer
      if (gesture & POINTER) == POINTER and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > leftMargin and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < 1-rightMargin \
              and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > topMargin and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < 1-bottomMargin:
        if abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - previousX) > (1-rightMargin-leftMargin)/180 and abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - previousY) > (1-topMargin-bottomMargin)/180:
            ag.moveTo((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x-leftMargin)*(1/(1-leftMargin-rightMargin))*screensize[0], (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y-topMargin)*(1/(1-topMargin-bottomMargin))*screensize[1], 0.1, ag.easeOutQuad)
            previousX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            previousY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

      if (gesture & SCROLL) == SCROLL:
          #ag.scroll(10)
          print(str((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)/(1-topMargin-bottomMargin)*200 ))
          ag.scroll(round((hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y)/(1-topMargin-bottomMargin)*400))

    cv2.putText(image, str(gesture) + " " + str(hand_unit),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()