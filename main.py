import math
import os

import cv2
import mediapipe as mp
import pyautogui as ag
import time
import subprocess as sp
import psutil

ag.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

screensize = ag.size()

leftMargin = 0.2
rightMargin = 0.2
topMargin = 0.1
bottomMargin = 0.6

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 100)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

LEFT_CLICK = 1
RIGHT_CLICK = 2
POINTER = 4
SCROLL = 8
PAUSE = 16
KEYBOARD = 32
MIDDLE_CLICK = 64


def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''
    listOfProcessObjects = []
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            # Check if process name contains the given name string.
            if processName.lower() in pinfo['name'].lower():
                listOfProcessObjects.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return listOfProcessObjects;


def distance(a, b) -> float:
    _x = a.x - b.x
    _y = a.y - b.y
    return math.sqrt(_x * _x + _y * _y)


def detectGesture(landmarks, unit=float) -> int:
    result = 0

    if distance(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.3 * unit and \
            distance(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]) < 0.4 * unit and \
            distance(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) > 0.6 * unit and \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_PIP].y:
        return PAUSE

    if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.PINKY_MCP].y:
        return KEYBOARD

    if distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) < 0.2 * unit and landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
            distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) > 0.3 * unit:
        result = result | LEFT_CLICK
    elif distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                  landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]) < 0.2 * unit and landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
            distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) > 0.3 * unit:
        result = result | RIGHT_CLICK
    elif distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                  landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) < 0.25 * unit:
        result = result | SCROLL
    elif distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                  landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.25 * unit:
        result = result | MIDDLE_CLICK

        # landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
    if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_MCP].y:
        result = result | POINTER
        """and \
                    landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmarks.landmark[
                mp_hands.HandLandmark.PINKY_MCP].y"""

    return result


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:
    hand_unit = 0.0
    previousINDEXX = 0
    previousINDEXY = 0
    previousINDEXMCPX = 0
    previousINDEXMCPY = 0
    previousPINKYMCPX = 0
    previousPINKYMCPY = 0
    previousWRISTX = 0
    previousWRISTY = 0
    leftPressed = False
    leftClickCount = 0
    rightPressed = False
    rightClickCount = 0
    middlePressed = False
    middleClickCount = 0
    paused = False
    ignoreCount = 0
    keyboardIgnore = 0
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

        # If a hand is detected - process the gestures
        if ignoreCount > 0:
            ignoreCount -= 1

        if keyboardIgnore > 0:
            keyboardIgnore -= 1

        if results.multi_hand_landmarks and ignoreCount == 0:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Calculate hand measurement unit if the hand is facing the camera
            if distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]) < 1.75 * distance(
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]) and distance(
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]) > 1.25 * distance(
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]):
                hand_unit = distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                     hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])

            gesture = detectGesture(hand_landmarks, hand_unit)

            # Change the pause state if the pause gesture is detected
            if (gesture & PAUSE) == PAUSE:
                paused = not paused
                ignoreCount = 30

            # process all the other gestures if not paused
            if not paused:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for "open/close keyboard" gesture
                if (gesture & KEYBOARD) == KEYBOARD and keyboardIgnore == 0:
                    list = findProcessIdByName("osk.exe")
                    if len(list) > 0:
                        for elem in list:
                            os.kill(elem['pid'], 9)
                    else:
                        keyboardApp = sp.Popen("osk.exe", shell=True)
                    keyboardIgnore = 30

                # Check for pointer
                if (gesture & POINTER) == POINTER and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - previousINDEXX) > (
                        1 - rightMargin - leftMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - previousINDEXY) > (
                        1 - topMargin - bottomMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x - previousINDEXMCPX) > (
                        1 - rightMargin - leftMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y - previousINDEXMCPY) > (
                        1 - topMargin - bottomMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x - previousPINKYMCPX) > (
                        1 - rightMargin - leftMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y - previousPINKYMCPY) > (
                        1 - topMargin - bottomMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - previousWRISTX) > (
                        1 - rightMargin - leftMargin) / 150 and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - previousWRISTY) > (
                        1 - topMargin - bottomMargin) / 150:
                    ag.moveTo((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - leftMargin) * (
                                1 / (1 - leftMargin - rightMargin)) * screensize[0],
                              (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - topMargin) * (
                                          1 / (1 - topMargin - bottomMargin)) * screensize[1])
                    previousINDEXX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                    previousINDEXY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    previousINDEXMCPX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    previousINDEXMCPY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                    previousPINKYMCPX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                    previousPINKYMCPY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
                    previousWRISTX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                    previousWRISTY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                # Check for left click
                if (gesture & LEFT_CLICK) == LEFT_CLICK:
                    leftClickCount += 1
                    if leftClickCount > 3 and not leftPressed:
                        ag.mouseDown(button='left')
                        leftPressed = True
                elif leftClickCount > 0:
                    if leftClickCount > 3 and leftPressed:
                        ag.mouseUp(button='left')
                        leftPressed = False
                    else:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
                            ag.doubleClick(button='left')
                        else:
                            ag.leftClick()
                    leftClickCount = 0

                # Check for right click
                if (gesture & RIGHT_CLICK) == RIGHT_CLICK:
                    rightClickCount += 1
                    if rightClickCount > 3 and not rightPressed:
                        ag.mouseDown(button='right')
                        rightPressed = True
                elif rightClickCount > 0:
                    if rightClickCount > 3 and rightPressed:
                        ag.mouseUp(button='right')
                        rightPressed = False
                    else:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
                            ag.doubleClick(button='right')
                        else:
                            ag.rightClick()
                    rightClickCount = 0

                # Check for middle click
                if (gesture & MIDDLE_CLICK) == MIDDLE_CLICK:
                    middleClickCount += 1
                    if middleClickCount > 3 and not middlePressed:
                        ag.mouseDown(button='middle')
                        middlePressed = True
                elif middleClickCount > 0:
                    if middleClickCount > 3 and middlePressed:
                        ag.mouseUp(button='middle')
                        middlePressed = False
                    else:
                        ag.middleClick()
                    middleClickCount = 0

                # Check for scrolling
                if (gesture & SCROLL) == SCROLL:
                    ag.scroll(round((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y -
                                     hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) / (
                                                1 - topMargin - bottomMargin) * 400))

        """cv2.putText(image, str(gesture) + " " + str(hand_unit),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)"""

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
