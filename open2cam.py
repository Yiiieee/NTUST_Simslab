import cv2
import time

CAM1_INDEX = 0
CAM2_INDEX = 2

WIDTH = 640
HEIGHT = 480
FPS = 30

cap1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(CAM2_INDEX, cv2.CAP_V4L2)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap1.set(cv2.CAP_PROP_FPS, FPS)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap2.set(cv2.CAP_PROP_FPS, FPS)

if not cap1.isOpened():
    print("Failed to open Camera 0")
    exit()

if not cap2.isOpened():
    print("Failed to open Camera 2")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

timestamp = time.strftime("%Y%m%d_%H%M%S")
out1 = cv2.VideoWriter(f"camera1_{timestamp}.mp4", fourcc, FPS, (WIDTH, HEIGHT))
out2 = cv2.VideoWriter(f"camera2_another_angle{timestamp}.mp4", fourcc, FPS, (WIDTH, HEIGHT))

print("Recording... press q to stop")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        print("Failed to read frame from Camera 1")
        break

    if not ret2:
        print("Failed to read frame from Camera 2")
        break

    out1.write(frame1)
    out2.write(frame2)

    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()

print("Recording finished")