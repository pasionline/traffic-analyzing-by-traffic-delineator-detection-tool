import argparse
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

# THIS SHOULD BE CHANGED FOR EVERY VIDEO
# Polygon coordinates can be checked on this link: https://roboflow.github.io/polygonzone/
# Corners should follow these rules:
# A: Left Upper Corner
# B: Right Upper Corner
# C: Right Bottom Corner
# D: Left Bottom Corner

SOURCE = np.array([
    [382, 456],  # A
    [846, 435],  # B
    [1742, 611],  # C
    [642, 711]  # D
])
# THIS SHOULD BE CHANGED FOR EVERY VIDEO
TARGET_WIDTH = 25
# THIS SHOULD BE CHANGED FOR EVERY VIDEO
TARGET_HEIGHT = 100

TARGET = np.array(
    [
        [0, 0],  # A
        [TARGET_WIDTH - 1, 0],  # B
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],  # C
        [0, TARGET_HEIGHT - 1]  # D
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Bachelorthesis Supervision'
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()


def calculate_distances(tracked_points):
    distances = {}
    keys = list(tracked_points.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            id1 = keys[i]
            id2 = keys[j]
            point1 = np.array(tracked_points[id1][-1])
            point2 = np.array(tracked_points[id2][-1])
            delta = np.linalg.norm(point1 - point2)
            distances[(id1, id2)] = delta
    return distances


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO("yolov10m.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        print(points)
        labels = []
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_end - coordinate_start)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
