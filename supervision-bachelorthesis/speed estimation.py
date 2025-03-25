#  OLD dont use

import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov10n.pt")
names = model.model.names

cap = cv2.VideoCapture("highway_test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#line_pts = [(0, h), (w, h)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    #tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0)
    video_writer.write(im0)

    cv2.imshow("Speed Estimation", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()