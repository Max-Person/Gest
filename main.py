import os
import cv2
import math
import psutil
import mediapipe as mp
import pyautogui as gui
import subprocess as sp

gui.FAILSAFE = False
screensize = gui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

top_margin = 0.1
left_margin = 0.2
right_margin = 0.2
bottom_margin = 0.6
dead_zone_fraction = 300
frames_for_push_mouse_button = 5

POINTER = 1
LEFT_CLICK = 2
RIGHT_CLICK = 4
MIDDLE_CLICK = 8
SCROLLING = 16
PAUSE = 32
KEYBOARD = 64


def find_process_id_by_name(process_name) -> list[dict[str, int]]:
    # Get a list of all the PIDs of a all the running process whose name contains
    # the given string process_name
    list_of_process: list[dict[str, int]] = []

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            process_info: dict[str, int] = proc.as_dict(attrs=['pid', 'name', 'create_time'])

            # Check if process name contains the given name string.
            if process_name.lower() in process_info['name'].lower():
                list_of_process.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return list_of_process


def distance(a, b) -> float:
    _x = a.x - b.x
    _y = a.y - b.y
    return math.sqrt(_x * _x + _y * _y)


def detect_gesture(landmarks, unit) -> int:
    result = 0

    if distance(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.3 * unit and \
            distance(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]) < 0.4 * unit and \
            distance(landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) > 0.6 * unit and \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
        return PAUSE

    if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > \
            landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and \
            landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > \
            landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
        return KEYBOARD

    if distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) < 0.2 * unit and \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
            distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) > 0.3 * unit:
        result = result | LEFT_CLICK

    elif distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                  landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]) < 0.2 * unit and \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
            landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
            distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                     landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) > 0.3 * unit:
        result = result | RIGHT_CLICK

    elif distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                  landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) < 0.25 * unit:
        result = result | SCROLLING

    elif distance(landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                  landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.25 * unit:
        result = result | MIDDLE_CLICK

    if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
            landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > \
            landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y:
        result = result | POINTER

    return result


cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    hand_unit = 0.0

    previous_index_finger_tip_x = 0
    previous_index_finger_tip_y = 0
    previous_index_finger_mcp_x = 0
    previous_index_finger_mcp_y = 0
    previous_pinky_mcp_x = 0
    previous_pinky_mcp_y = 0
    previous_wrist_x = 0
    previous_wrist_y = 0

    left_button_pressed = False
    left_click_frames_count = 0

    right_button_pressed = False
    right_click_frames_count = 0

    middle_button_pressed = False
    middle_click_frames_count = 0

    paused = False

    ignore_frames_count = 0
    keyboard_ignore_frames_count = 0

    while cap.isOpened():
        gesture = 0
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
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
        if ignore_frames_count > 0:
            ignore_frames_count -= 1

        if keyboard_ignore_frames_count > 0:
            keyboard_ignore_frames_count -= 1

        if results.multi_hand_landmarks and ignore_frames_count == 0:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Calculate hand measurement unit if the hand is facing the camera
            if distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]) < \
                    1.85 * distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]) and \
                    distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]) > \
                    1.25 * distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]):

                hand_unit = distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                     hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])

                gesture = detect_gesture(hand_landmarks, hand_unit)

                # Change the pause state if the pause gesture is detected
                if (gesture & PAUSE) == PAUSE:
                    paused = not paused
                    ignore_frames_count = 30

                # process all the other gestures if not paused
                if not paused:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Check for "open/close keyboard" gesture
                    if (gesture & KEYBOARD) == KEYBOARD and keyboard_ignore_frames_count == 0:
                        list_of_process_objects: list[dict[str, int]] = find_process_id_by_name("osk.exe")

                        if len(list_of_process_objects) > 0:
                            for elem in list_of_process_objects:
                                os.kill(elem['pid'], 9)
                        else:
                            try:
                                keyboardApp = sp.Popen("osk.exe", shell=True)
                            except():
                                print("Can't kill osk.exe: ")
                        keyboard_ignore_frames_count = 30

                    # Check for pointer
                    if (gesture & POINTER) == POINTER and abs(
                            hand_landmarks.landmark[
                                mp_hands.HandLandmark.INDEX_FINGER_TIP].x - previous_index_finger_tip_x) > (
                            1 - right_margin - left_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP].y - previous_index_finger_tip_y) > (
                            1 - top_margin - bottom_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].x - previous_index_finger_mcp_x) > (
                            1 - right_margin - left_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP].y - previous_index_finger_mcp_y) > (
                            1 - top_margin - bottom_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x - previous_pinky_mcp_x) > (
                            1 - right_margin - left_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y - previous_pinky_mcp_y) > (
                            1 - top_margin - bottom_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - previous_wrist_x) > (
                            1 - right_margin - left_margin) / dead_zone_fraction and abs(
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - previous_wrist_y) > (
                            1 - top_margin - bottom_margin) / dead_zone_fraction:

                        gui.moveTo((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - left_margin) * (
                                1 / (1 - left_margin - right_margin)) * screensize[0],
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - top_margin) * (
                                          1 / (1 - top_margin - bottom_margin)) * screensize[1])

                        previous_index_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                        previous_index_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                        previous_index_finger_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                        previous_index_finger_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                        previous_pinky_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                        previous_pinky_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
                        previous_wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                        previous_wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                    # Check for left click
                    if (gesture & LEFT_CLICK) == LEFT_CLICK:
                        left_click_frames_count += 1
                        if left_click_frames_count > frames_for_push_mouse_button and not left_button_pressed:
                            gui.mouseDown(button='left')
                            left_button_pressed = True
                    elif left_click_frames_count > 0:
                        if left_click_frames_count > frames_for_push_mouse_button and left_button_pressed:
                            gui.mouseUp(button='left')
                            left_button_pressed = False
                        else:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < \
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
                                gui.doubleClick(button='left')
                            else:
                                gui.leftClick()
                        left_click_frames_count = 0

                    # Check for right click
                    if (gesture & RIGHT_CLICK) == RIGHT_CLICK:
                        right_click_frames_count += 1
                        if right_click_frames_count > frames_for_push_mouse_button and not right_button_pressed:
                            gui.mouseDown(button='right')
                            right_button_pressed = True
                    elif right_click_frames_count > 0:
                        if right_click_frames_count > frames_for_push_mouse_button and right_button_pressed:
                            gui.mouseUp(button='right')
                            right_button_pressed = False
                        else:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < \
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
                                gui.doubleClick(button='right')
                            else:
                                gui.rightClick()
                        right_click_frames_count = 0

                    # Check for middle click
                    if (gesture & MIDDLE_CLICK) == MIDDLE_CLICK:
                        middle_click_frames_count += 1
                        if middle_click_frames_count > frames_for_push_mouse_button and not middle_button_pressed:
                            gui.mouseDown(button='middle')
                            middle_button_pressed = True
                    elif middle_click_frames_count > 0:
                        if middle_click_frames_count > frames_for_push_mouse_button and middle_button_pressed:
                            gui.mouseUp(button='middle')
                            middle_button_pressed = False
                        else:
                            gui.middleClick()
                        middle_click_frames_count = 0

                    # Check for scrolling
                    if (gesture & SCROLLING) == SCROLLING:
                        gui.scroll(round((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y -
                                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) / (
                                                1 - top_margin - bottom_margin) * 400))

        cv2.imshow("Webcam", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
