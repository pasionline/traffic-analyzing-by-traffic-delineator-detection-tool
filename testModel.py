import cv2
from ultralytics import YOLOv10
import supervision as sv

model = YOLOv10(f'bestv3.pt')
results = model(source=f'test.jpeg', conf=0.25)
print(results[0].boxes.xyxy)


detections = sv.Detections.from_ultralytics(results[0])

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
image = cv2.imread('test.jpeg')
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
