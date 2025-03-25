import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

model = YOLO(f'best_250210.pt')
frame_generator = sv.get_video_frames_generator('typicalGermanAutobahn.mkv')

image = next(frame_generator)

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(callback=callback)
detections = slicer(image)

#i = 0
#while i < 50:
#    image = next(frame_generator)
#    nextDetections = slicer(image)
#    if len(nextDetections.xyxy) > len(detections.xyxy):
#        detections = nextDetections
#    i += 1

print(detections.xyxy)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
