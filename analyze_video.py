import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO
import supervision as sv
import csv
import argparse
from typing import List, Tuple

TARGET_WIDTH = 50
TARGET_HEIGHT = 100
# Percentage Range in which the cars are tracked
TARGET_DELTA = 5

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
    def __init__(self, tracking_id, start_frame=None, end_frame=None, speed=None, start_to_mid_speed=None,
                 mid_to_end_speed=None, acceleration=None, first_crossed=None, middle_frame=None):
        self.tracking_id = tracking_id
        self.start_frame = start_frame  # frame_count when passed the first gate
        self.middle_frame = middle_frame
        self.end_frame = end_frame  # frame_count when passed the second gate
        self.speed = speed
        self.start_to_mid_speed = start_to_mid_speed
        self.mid_to_end_speed = mid_to_end_speed
        self.acceleration = acceleration
        self.first_crossed = first_crossed  # "top" or "bottom"
        self.vertical_average_distances = {}  # {other_id: averaged distance}


# Returns point array of detected posts using a yolo model
def get_coordinates(video_path) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    model = YOLO(f'best_YOLOv12_traffic-delinator.pt')
    frame_gen = sv.get_video_frames_generator(video_path)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)

    image = next(frame_gen)
    slicer = sv.InferenceSlicer(callback=callback)
    detections = slicer(image)

    # Rotating the first couple of frames in case a Vehicle blocks the vision on a delineator
    i = 0
    while i < 20:
        image = next(frame_gen)

        nextDetections = slicer(image)
        if len(nextDetections.xyxy) > len(detections.xyxy):
            detections = nextDetections
        next(frame_gen)
        next(frame_gen)
        i += 1

    # Plotting the detection:
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(
         scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
           scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)

    coords = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

    if len(coords) < 4:
        print("Not enough points to cluster.")
        exit(-1)

    # Cluster into two groups
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
    labels = kmeans.labels_
    cluster_0 = coords[labels == 0]
    cluster_1 = coords[labels == 1]

    # Fit lines to each cluster
    def fit_line(points):
        x = points[:, 0]
        y = points[:, 1]
        m, b = np.polyfit(x, y, 1)
        return m, b

    m0, b0 = fit_line(cluster_0)
    m1, b1 = fit_line(cluster_1)

    # Draw lines for visualization
    img_with_lines = image.copy()

    def draw_line(img, m, b, color):
        h, w = img.shape[:2]
        x1, y1 = 0, int(b)
        x2, y2 = w, int(m * w + b)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    draw_line(img_with_lines, m0, b0, (255, 0, 0))  # blue
    draw_line(img_with_lines, m1, b1, (0, 255, 0))  # green

    for (x, y), label in zip(coords, labels):
        cv2.circle(img_with_lines, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(img_with_lines, str(label), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detected Lines", img_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # sort clusters by y-coord
    cluster_0 = cluster_0[np.argsort(cluster_0[:, 1])[::-1]]
    cluster_1 = cluster_1[np.argsort(cluster_1[:, 1])[::-1]]
    print(cluster_0, cluster_1)
    # pair them:
    paired_delineators = list(zip(cluster_0, cluster_1))

    return np.array(coords), paired_delineators


def crossing_gate(data, gate, frame_counter, distance):
    half_distance = distance / 2
    opposite_gate = ""
    if gate == "bottom":
        opposite_gate = "top"
    elif gate == "top":
        opposite_gate = "bottom"
    elif gate == "middle":
        # vehicle has to be tracked during the first gate
        if data is not None:
            data.middle_frame = frame_counter
        return
    else:
        raise Exception(f"Unknown gate: {gate}")

    if data is None:
        # First time seeing the vehicle
        vehicle_data[tracker_id] = VehicleData(tracker_id, start_frame=frame_counter, first_crossed=gate)
    elif data.first_crossed == opposite_gate and data.end_frame is None:
        # Already crossed one line before; now completing second crossing
        data.end_frame = frame_counter
        frame_diff_start_to_end = abs(data.end_frame - data.start_frame)

        time_start_end = frame_diff_start_to_end / video_info.fps
        data.speed = (distance / time_start_end) * 3.6

        if data.middle_frame is not None:
            frame_diff_start_to_mid = abs(data.middle_frame - data.start_frame)
            frame_diff_mid_to_end = abs(data.end_frame - data.middle_frame)

            time_start_mid = frame_diff_start_to_mid / video_info.fps
            time_mid_end = frame_diff_mid_to_end / video_info.fps

            data.start_to_mid_speed = (half_distance / time_start_mid) * 3.6
            data.mid_to_end_speed = (half_distance / time_mid_end) * 3.6
            data.acceleration = (data.mid_to_end_speed - data.start_to_mid_speed) / time_mid_end

            # average distances
            avg_distances = {}
            for other_id in distance_sums.get(tracker_id, {}):
                total = distance_sums[tracker_id][other_id]
                count = distance_counts[tracker_id][other_id]
                avg = total / count
                avg_distances[other_id] = round(avg, 2)

            avg_d_clean = {int(k): float(v) for k, v in avg_distances.items()}
            data.vertical_average_distances = avg_d_clean
            print(
                f"[{gate}] Vehicle {tracker_id} totalspeed: {data.speed:.2f} km/h midspeed: {data.start_to_mid_speed} endspeed: {data.mid_to_end_speed} acc:{data.acceleration:.2f}")
        else:
            # mid was not tracked:
            print(f"[{gate}] Vehicle {tracker_id} totalspeed: {data.speed:.2f} km/h")


if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.parse_args()

    parser.add_argument("-p", "--path", type=str, help="Relative path to video file")
    parser.add_argument("-d", "--distance", type=str, help="Distance between two delineators (in meters, default: 50m)")
    parser.add_argument("-m", "--model", type=str,
                        help="Optional: YOLO model used for vehicle detection (e.g., 'yolo12l.pt')")
    parser.add_argument("--plot", type=str, help="Enables plotting for debugging or visualization")

    args = parser.parse_args()

    videoPath = args.path if args.path else "example-videos/example.mp4"
    DISTANCE = args.distance if args.distance else 50.0  # default 50 meters
    modelType = args.model if args.model else "yolo12l.pt"
    plot = True if args.plot else False

    video_info = sv.VideoInfo.from_video_path(videoPath)
    model = YOLO(modelType)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Get video infos
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    unsorted_coords, paired_delineators = get_coordinates(videoPath)

    # Corners follow these rules:
    # A: Left Upper Corner
    # B: Right Upper Corner
    # C: Right Bottom Corner
    # D: Left Bottom Corner

    sections = []
    for i in range(len(paired_delineators) - 1):
        bottom_pair = paired_delineators[i]
        top_pair = paired_delineators[i + 1]

        section = {
            "top_0": np.round(top_pair[0]),
            "top_1": np.round(top_pair[1]),
            "bottom_0": np.round(bottom_pair[0]),
            "bottom_1": np.round(bottom_pair[1])
        }

        sections.append(section)

    source = np.array([sections[0]["top_0"],  # A
                       sections[0]["top_1"],  # B
                       sections[0]["bottom_1"],  # C
                       sections[0]["bottom_0"]]  # D
                      , dtype="int32")
    print(source)


    polygon_zone = sv.PolygonZone(source)
    view_transformer = ViewTransformer(source=source)

    # calculate speed estimation range
    top_delta = TARGET_DELTA
    bottom_delta = TARGET_HEIGHT - TARGET_DELTA
    mid_delta1 = (TARGET_HEIGHT / 2) - (TARGET_DELTA / 2)
    mid_delta2 = (TARGET_HEIGHT / 2) + (TARGET_DELTA / 2)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(videoPath)

    frame_counter = 0

    # Vehicle Data stores all the collected data of each vehicle, accessed through tracker_id
    vehicle_data = {}
    #  Temporary data used to calculate and stored to vehicle_data later
    ## Distances
    distance_sums = {}  # {tracker_id: {other_id: sum}}
    distance_counts = {}  # {tracker_id: {other_id: count}}

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
            if bottom_delta < y < TARGET_HEIGHT:
                crossing_gate(data=data, gate="bottom", frame_counter=frame_counter, distance=DISTANCE)

            # Case 2: Vehicle crosses top threshold (delta1 to 0)
            elif 0 < y < top_delta:
                crossing_gate(data=data, gate="top", frame_counter=frame_counter, distance=DISTANCE)
            elif mid_delta1 < y < mid_delta2:
                crossing_gate(data=data, gate="middle", frame_counter=frame_counter, distance=DISTANCE)

            # Distance Estimation:
            if tracker_id not in distance_sums:
                distance_sums[tracker_id] = {}
                distance_counts[tracker_id] = {}

            for other_id, [_, other_y] in zip(detections.tracker_id, vehicle_target_coords):
                other_data = vehicle_data.get(other_id)
                if other_data is None or data is None:
                    continue
                # skip same vehicle
                if other_id == tracker_id and other_data.first_crossed != data.first_crossed:
                    continue

                scale_factor = DISTANCE / TARGET_HEIGHT  # meters per pixel
                vertical_distance = abs(y - other_y) * scale_factor

                # Update sum and count
                distance_sums[tracker_id][other_id] = distance_sums[tracker_id].get(other_id, 0) + vertical_distance
                distance_counts[tracker_id][other_id] = distance_counts[tracker_id].get(other_id, 0) + 1

        # Label each vehicle:
        labels = []
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            data = vehicle_data.get(tracker_id)
            speed = f"{data.speed:.2f}" if data and data.speed is not None else "N/A"
            label = f"#{tracker_id} {result.names[class_id]} {confidence:.2f} Speed: {speed} km/h"
            labels.append(label)

        # Annotated Frame
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=source, color=sv.Color.RED)
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
        top_down_scaled = cv2.resize(top_down_frame, (TARGET_WIDTH * scale_factor, TARGET_HEIGHT * scale_factor),
                                     interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Top-down Warped Frame", top_down_scaled)

    cv2.destroyAllWindows()

    # CSV Export
    with open("vehicle_data.csv", mode="w", newline="") as csv_file:
        fieldnames = [
            "tracking_id",
            "start_frame",
            "middle_frame",
            "end_frame",
            "first_crossed",
            "speed",
            "start_to_mid_speed",
            "mid_to_end_speed",
            "acceleration",
            "average_vertical_distances",
            "valid"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for data in vehicle_data.values():
            writer.writerow({
                "tracking_id": data.tracking_id,
                "start_frame": data.start_frame,
                "middle_frame": data.middle_frame,
                "end_frame": data.end_frame,
                "first_crossed": data.first_crossed,
                "speed": round(data.speed, 2) if data.speed else None,
                "start_to_mid_speed": round(data.start_to_mid_speed, 2) if data.start_to_mid_speed else None,
                "mid_to_end_speed": round(data.mid_to_end_speed, 2) if data.mid_to_end_speed else None,
                "acceleration": round(data.acceleration, 2) if data.acceleration else None,
                "average_vertical_distances": data.vertical_average_distances,
                "valid": bool(data.end_frame is not None)
            })
