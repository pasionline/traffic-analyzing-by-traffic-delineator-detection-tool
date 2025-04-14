import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Corners follow these rules:
# A: Left Upper Corner
# B: Right Upper Corner
# C: Right Bottom Corner
# D: Left Bottom Corner

TARGET_WIDTH = 25
TARGET_HEIGHT = 100
# Range in which the cars are tracked
TARGET_DELTA = 5

DISTANCE = 100 # difference in meters between two delineators

TARGET = np.array(
    [
        [0, 0],  # A
        [TARGET_WIDTH - 1, 0],  # B
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],  # C
        [0, TARGET_HEIGHT - 1]  # D
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray) -> None:
        target = np.array(TARGET)

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class VehicleData:
    def __init__(self, tracking_id, start_frame=None, end_frame=None, speed=None, first_crossed=None):
        self.tracking_id = tracking_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.speed = speed
        self.first_crossed = first_crossed  # "top" or "bottom"


# Returns point array of detected posts using a yolo model
def get_coordinates(video_path) -> np.ndarray:
    model = YOLO(f'best_250210.pt')
    frame_gen = sv.get_video_frames_generator(video_path)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)

    image = next(frame_gen)

    slicer = sv.InferenceSlicer(callback=callback)
    detections = slicer(image)

    # Rotating some frames, if vehicles blocking the view (uncommented bc it has a long runtime)
    # i = 0
    # while i < 50:
    #    image = next(frame_generator)
    #    nextDetections = slicer(image)
    #    if len(nextDetections.xyxy) > len(detections.xyxy):
    #        detections = nextDetections
    #    i += 1

    # Plotting the detection:
    # box_annotator = sv.BoxAnnotator()
    # label_annotator = sv.LabelAnnotator()

    # annotated_image = box_annotator.annotate(
    #     scene=image, detections=detections)
    #  annotated_image = label_annotator.annotate(
    #       scene=annotated_image, detections=detections)
    #
    # sv.plot_image(annotated_image)

    return detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)


if __name__ == "__main__":

    videoPath = "highway_test.mp4"

    video_info = sv.VideoInfo.from_video_path(videoPath)
    model = YOLO("yolo11l.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Get video infos
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    # raw_source = get_coordinates("vehicles.mp4")
    # TODO: transform raw source in something reliable

    raw_source = np.array([
        [382, 456],     # A
        [846, 435],     # B
        [1742, 611],    # C
        [642, 711]      # D
    ])

    source = []
    polygon_zone = sv.PolygonZone(raw_source)
    view_transformer = ViewTransformer(source=raw_source)

    # calculate speed estimation range
    delta1 = TARGET_DELTA
    delta2 = TARGET_HEIGHT - TARGET_DELTA

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(videoPath)

    frame_counter = 0
    # Vehicle Data stores all the collected data of each vehicle, accessed through tracker_id
    vehicle_data = {}
    # Process Video:
    for frame in frame_generator:

        result = model(frame, verbose=False)[0]
        # Object detection
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        # Transform Picture coords into coords without perspective disturbance (top-down perspective)
        vehicle_source_coords = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        vehicle_target_coords = view_transformer.transform_points(vehicle_source_coords)

        # print(vehicle_target_coords)

        for tracker_id, [x, y] in zip(detections.tracker_id, vehicle_target_coords):
            data = vehicle_data.get(tracker_id)

        # Speed Estimation:
            # Case 1: Vehicle crosses bottom threshold (delta2 to TARGET_HEIGHT)
            if delta2 < y < TARGET_HEIGHT:
                if data is None:
                    # First time seeing the vehicle
                    vehicle_data[tracker_id] = VehicleData(tracker_id, start_frame=frame_counter, first_crossed="bottom")
                elif data.first_crossed == "top" and data.end_frame is None:
                    # Already crossed one line before; now completing second crossing
                    data.end_frame = frame_counter
                    frame_diff = abs(data.end_frame - data.start_frame)
                    time = frame_diff / video_info.fps
                    data.speed = (DISTANCE / time) * 3.6  # 50 meters is the known distance
                    print(f"[top cross] Vehicle {tracker_id} speed: {data.speed:.2f} km/h")

            # Case 2: Vehicle crosses top threshold (delta1 to 0)
            elif 0 < y < delta1:
                if data is None:
                    # First time seeing the vehicle
                    vehicle_data[tracker_id] = VehicleData(tracker_id, start_frame=frame_counter, first_crossed="top")
                elif data.first_crossed == "bottom" and data.end_frame is None:
                    # Already crossed one line before; now completing second crossing
                    data.end_frame = frame_counter
                    frame_diff = abs(data.end_frame - data.start_frame)
                    time = frame_diff / video_info.fps
                    data.speed = (DISTANCE / time) * 3.6
                    print(f"[bottom cross] Vehicle {tracker_id} speed: {data.speed:.2f} km/h")



        # Label each vehicle:
        labels = []
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            data = vehicle_data.get(tracker_id)
            speed = f"{data.speed:.2f}" if data and data.speed is not None else "N/A"
            label = f"#{tracker_id} {result.names[class_id]} {confidence:.2f} Speed: {speed} km/h"
            labels.append(label)

        # Annotated Frame
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=raw_source, color=sv.Color.RED)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        frame_counter += 1

        ## Plotting:
        # Standard view:
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break

        # Top-down view:
        scale_factor = 10
        warped_size = (TARGET_WIDTH, TARGET_HEIGHT)
        top_down_frame = cv2.warpPerspective(frame, view_transformer.m, warped_size)
        top_down_scaled = cv2.resize(top_down_frame, (TARGET_WIDTH * scale_factor, TARGET_HEIGHT * scale_factor), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Top-down Warped Frame", top_down_scaled)

    cv2.destroyAllWindows()
