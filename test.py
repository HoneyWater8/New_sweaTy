import modules
import cv2
import pandas as pd
import mediapipe as mp
import time
import asyncio
import aiohttp
import json
import os
import ssl

url = 'https://suheonchoi-azureml.eastus.inference.ml.azure.com/score'
api_key = 'xX20kmKXq1vKZDTINHkLkB8dpEgj5RdI'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
async def send_request(session, url, headers, data):
    body = str.encode(json.dumps(data))
    async with session.post(url, headers=headers, data=body) as response:
        result = int((await response.text())[1:-1])
        return result

async def play_2D_point(test_video_path):
    # Initialize the VideoCapture object to read from a video stored on disk.
    video = cv2.VideoCapture(test_video_path)

    # Initializing mediapipe pose class.
    mp_pose = mp.solutions.pose

    # Initializing mediapipe drawing class, useful for annotation.
    mp_drawing = mp.solutions.drawing_utils

    # pose detection function start
    pose_video = mp_pose.Pose(static_image_mode=False,
                              min_tracking_confidence=0.1,
                              min_detection_confidence=0.8,
                              model_complexity=1,
                              smooth_landmarks=True)

    # Variables for squat counting
    squatCnt = 0
    squatBeforeState = 1  # initial state (standing)
    squatNowState = 1  # initial state (standing)
    squatState = []  # squat state record (correct posture: 1, incorrect posture: 0)
    squatAccuracy = -1  # initial accuracy
    stateQueue = [-1, -1, -1, -1, -1]
    class_idx_adj = -1

    paused = False
    i = 0

    # Prepare aiohttp session
    async with aiohttp.ClientSession() as session:
        while video.isOpened():
            i += 1
            # Read a frame.
            hasFrame, frame = video.read()

            # Check if frame is not read properly.
            if not hasFrame:
                break

            # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)

            # Get the width and height of the frame
            frame_height, frame_width, _ = frame.shape

            # Resize the frame while keeping the aspect ratio.
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # Perform pose detection
            results, frame, landmarks = modules.detectPose(frame, pose_video, mp_pose, mp_drawing, display=False)

            # Check if all landmarks are detected
            if results.pose_world_landmarks is not None and all(
                    results.pose_world_landmarks.landmark[j].visibility > 0.1 for j in [11, 12, 23, 24, 25, 26, 27, 28]):
                cv2.putText(frame, "All landmarks detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Normalize joint coordinates
                adjust_x = -1 * landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0]
                adjust_y = -1 * landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]

                landmarks_adjust_point = [(landmarks[j][0] + adjust_x, landmarks[j][1] + adjust_y) for j in range(33)]

                hip_distance = modules.calculateDistance2D(
                    landmarks_adjust_point[mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks_adjust_point[mp_pose.PoseLandmark.RIGHT_HIP.value]
                )

                landmarks_adjust_ratio = [(landmarks_adjust_point[j][0] / hip_distance,
                                           landmarks_adjust_point[j][1] / hip_distance) for j in range(33)]

                # Prepare data for request
                data = {
                    "input_data": {
                        "columns": [
                            "right_shoulder_x", "right_shoulder_y",
                            "left_shoulder_x", "left_shoulder_y",
                            "right_hip_x", "right_hip_y",
                            "left_hip_x", "left_hip_y",
                            "right_knee_x", "right_knee_y",
                            "left_knee_x", "left_knee_y",
                            "right_ankle_x", "right_ankle_y",
                            "left_ankle_x", "left_ankle_y"
                        ],
                        "index": [0],
                        "data": [
                            [round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_HIP.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_HIP.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_HIP.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_HIP.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_KNEE.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_KNEE.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_KNEE.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_KNEE.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_ANKLE.value][0], 2),
                             round(landmarks_adjust_ratio[mp_pose.PoseLandmark.LEFT_ANKLE.value][1], 2)]
                        ]
                    }
                }

                # Send request asynchronously
                start_time = time.time()
                try:
                    result = await send_request(session, url, headers, data)
                    print(result)
                    print(type(result))
                except aiohttp.ClientError as error:
                    print("The request failed: " + str(error))
                end_time = time.time()

                # Execution time calculation
                execution_time = end_time - start_time
                print(f"코드 실행 시간: {execution_time:.6f} 초")

                class_list = ['good_stand', 'good_progress', 'good_sit',
                              'knee_narrow_progress', 'knee_narrow_sit',
                              'knee_wide_progress', 'knee_wide_sit']
                class_idx = result

                # Update state queue
                stateQueue.pop(0)
                stateQueue.append(class_idx)

                # Check if all states are the same
                all_same = all(element == stateQueue[0] for element in stateQueue)

                if all_same:
                    class_idx_adj = stateQueue[0]

                if class_idx_adj == -1:
                    cv2.putText(frame, f"Class: Waiting...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif class_idx_adj == 0:
                    cv2.putText(frame, f"Class: {class_list[class_idx_adj]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif class_idx_adj < 3:
                    squatState.append(1)
                    cv2.putText(frame, f"Class: {class_list[class_idx_adj]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    squatState.append(0)
                    cv2.putText(frame, f"Class: {class_list[class_idx_adj]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Determine squat states and update count
                if class_idx_adj == 0:
                    squatNowState = 1
                elif class_idx_adj == 2 or class_idx_adj == 4 or class_idx_adj == 6:
                    squatNowState = 0

                if squatBeforeState == 0 and squatNowState == 1:
                    squatAccuracy = sum(squatState) / len(squatState)

                    if squatAccuracy > 0.7:
                        squatCnt += 1

                    squatState = []

                squatBeforeState = squatNowState

                # Display squat accuracy
                if squatAccuracy >= 0.7:
                    cv2.putText(frame, f"{squatAccuracy * 100:.1f}%", (360, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, f"{squatAccuracy * 100:.1f}%", (360, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            else:
                cv2.putText(frame, "Not all landmarks detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"Frame: {i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Display the frame.
            cv2.imshow('Mediapipe Pose', frame)

            # Exit if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the VideoCapture object.
    video.release()
    cv2.destroyAllWindows()


test_video_path = "test3_2.mp4"
# Run the play_2D_point function as an asyncio task
asyncio.run(play_2D_point(test_video_path))
