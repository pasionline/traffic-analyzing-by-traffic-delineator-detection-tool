import cv2
from ultralytics import YOLOv10
import supervision as sv

model = YOLOv10(f'bestv6.pt')
frame_generator = sv.get_video_frames_generator('crash_test.mp4')
image = next(frame_generator)

results = model(source=image, conf=0.25)
print(results[0].boxes.xyxy)


detections = sv.Detections.from_ultralytics(results[0])

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
