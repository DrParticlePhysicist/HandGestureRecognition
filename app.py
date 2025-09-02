#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    return args


def main():
    # Args
    args = get_args()
    cap_device, cap_width, cap_height = args.device, args.width, args.height

    # Camera
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Classifier (safe load: only if model exists & not empty)
    keypoint_classifier = None
    tflite_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
    if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 10:
        keypoint_classifier = KeyPointClassifier(model_path=tflite_path)
    else:
        print("⚠️ No trained model found. Inference disabled, only logging mode available.")

    # Labels
    label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    if os.path.exists(label_path):
        with open(label_path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    else:
        keypoint_classifier_labels = []

    # Utils
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # History
    point_history = deque(maxlen=16)

    # Modes
    mode = 0   # 0 → inference, 1 → logging
    number = -1

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)

        # ESC → exit
        if key == 27:
            break

        # Mode selection
        number, mode = select_mode(key, mode)

        # Capture frame
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark list
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # Preprocess
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Logging (dataset collection)
                logging_csv(number, mode, pre_processed_landmark_list)

                # Inference
                if mode == 0 and keypoint_classifier is not None:
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                elif mode == 1:
                    hand_sign_text = "Logging..."
                else:
                    hand_sign_text = "No model"

                # Draw
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_text)

        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


# ---------- Helper Functions ----------
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0–9
        number = key - 48
    if key == 107:  # 'k' key → toggle logging/inference
        mode = 1 - mode
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] 
                               for landmark in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark]


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for point in temp_landmark_list:
        point[0] -= base_x
        point[1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return [n / max_value for n in temp_landmark_list]


def logging_csv(number, mode, landmark_list):
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text += ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    if mode == 1:
        cv.putText(image, "MODE: Logging", (10, 60), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 80), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1, cv.LINE_AA)
    else:
        cv.putText(image, "MODE: Inference", (10, 60), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
