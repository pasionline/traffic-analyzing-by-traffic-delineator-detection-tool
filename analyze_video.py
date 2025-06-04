import os
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from ultralytics import YOLO
import supervision as sv
import csv
import argparse
from typing import Any
import glob

TARGET_WIDTH = 75
TARGET_HEIGHT = 150
# Percentage Range in which the cars are tracked
TARGET_DELTA = 0.05 * TARGET_HEIGHT

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
def get_coordinates(video_path, calibration, plot) -> list[tuple[Any, Any]]:
    model = YOLO(f'best_YOLOv12_traffic-delinator.pt')
    frame_gen = sv.get_video_frames_generator(video_path)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, conf=0.2)[0]
        return sv.Detections.from_ultralytics(result)

    image = next(frame_gen)
    slicer = sv.InferenceSlicer(callback=callback)
    detections = slicer(image)

    # Rotating the first couple of frames in case a Vehicle blocks the vision on a delineator
    i = 0
    while i < 5:
        image = next(frame_gen)

        nextDetections = slicer(image)
        if len(nextDetections.xyxy) > len(detections.xyxy):
            detections = nextDetections
        j = 0
        while (j < 5):
            next(frame_gen)
            j += 1
        i += 1

    coords = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

    if plot:
        # Annotate detections
        annotator = sv.BoxAnnotator()
        annotated_image = annotator.annotate(scene=image.copy(), detections=detections)

        # Display the image (optional, e.g., in Jupyter or local test)
        cv2.imshow("Detections", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Suppose coords is your (N,2) array of points
    coords = np.array(coords, dtype=np.int32)
    coords = filter_close_points(coords, delta=50)

    if (calibration == "houghline"):
        above, below = find_pairs_hough_line_transform(detections, coords, image, plot)
    else:
        # use ransac if user typed anything
        above, below = find_pairs_ransac(coords, image, plot)

    above = above[np.argsort(above[:, 1])[::-1]]
    below = below[np.argsort(below[:, 1])[::-1]]

    # pair them:
    paired_delineators = list(zip(above, below))

    return paired_delineators


def find_pairs_ransac(coords, image, plot):
    # get X and Y in the correct format for ransac
    X = coords[:, 0].reshape(-1, 1)
    y = coords[:, 1]

    lines12 = []
    # First RANSAC Line
    ransac1, inlier_mask1, outlier_mask1 = find_best_line(X, y)

    if ransac1:
        X_line1 = X[inlier_mask1]
        y_line1 = y[inlier_mask1]
        lines12.append(np.hstack((X_line1, y_line1.reshape(-1, 1))))

        X_remaining = X[outlier_mask1]
        y_remaining = y[outlier_mask1]

        # Second RANSAC Line
        ransac2, inlier_mask2, outlier_mask2 = find_best_line(X_remaining, y_remaining)

        if ransac2:
            X_line2 = X_remaining[inlier_mask2]
            y_line2 = y_remaining[inlier_mask2]
            X_noise = X_remaining[outlier_mask2]
            y_noise = y_remaining[outlier_mask2]
            lines12.append(np.hstack((X_line2, y_line2.reshape(-1, 1))))
        else:
            X_line2, y_line2 = np.array([]), np.array([])
            X_noise, y_noise = X_remaining, y_remaining
    else:
        print("No Lines detected")
        X_line1, y_line1, X_line2, y_line2, X_noise, y_noise = [np.array([])] * 6

    # re-array all points in their category
    line1 = np.hstack((X_line1, y_line1.reshape(-1, 1)))
    line2 = np.hstack((X_line2, y_line2.reshape(-1, 1)))
    noise = np.hstack((X_noise, y_noise.reshape(-1, 1)))
    if plot:
        # Plot points:
        for pt in line1:
            cv2.circle(image, tuple(pt.astype(int)), 6, (100, 149, 237), -1)  # cornflowerblue
        for pt in line2:
            cv2.circle(image, tuple(pt.astype(int)), 6, (50, 205, 50), -1)  # limegreen
        for pt in noise:
            cv2.drawMarker(image, tuple(pt.astype(int)), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)

        # Plot RANSAC Line
        def draw_line_on_image(ransac_model, X_data, img, color):
            if ransac_model:
                line_X = np.linspace(X_data.min(), X_data.max(), 100).reshape(-1, 1)
                line_y = ransac_model.predict(line_X)
                points = np.vstack((line_X.flatten(), line_y)).T.astype(np.int32)
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    cv2.line(img, pt1, pt2, color, thickness=2)

        draw_line_on_image(ransac1, X, image, (255, 0, 0))  # Blau
        draw_line_on_image(ransac2, X, image, (0, 255, 0))  # Grün

        # Speichern
        cv2.imshow("Output", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return line1, line2


def find_best_line(X_data, y_data):
    # RANSAC helper function
    if len(X_data) < 2:
        return None, None, None

    ransac = RANSACRegressor(
        min_samples=2,  # Mindestpunkte für eine Linie
        residual_threshold=15.0,  # Max. Abstand zur Linie (Pixel), um Inlier zu sein
        max_trials=1000  # Anzahl der Versuche
    )
    ransac.fit(X_data, y_data)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    return ransac, inlier_mask, outlier_mask


def find_pairs_hough_line_transform(detections, coords, image, plot):
    # THIS METHOD IS USED TO FIND WHITE EDGES AND USE THE MEDIAN LINE TO SEPERATE LEFT AND RIGHT
    print(coords)
    if len(detections.xyxy) < 4:
        raise Exception('Not enough detections to form a region of interest')

    # Compute the convex hull (returns ordered points)
    hull = cv2.convexHull(coords)

    # Convert image to HSV colorspace
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Define range of colors in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 20, 255])  # Threshold the HSV image to get only blue colors
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    color_isolated = cv2.bitwise_and(image, image, mask=white_mask)
    # Convert to Grayscale
    gray = cv2.cvtColor(color_isolated, cv2.COLOR_RGB2GRAY)  # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Perform edge detection
    edges = cv2.Canny(blur_gray, 50, 150)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # Do the Masking
    cv2.fillPoly(mask, [hull], ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    if len(lines) <= 0:
        raise Exception('No lines detected')

    # Iterate over the output "lines" to calculate m's and b's
    mxb = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:  # avoid division by zero
                continue
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            mxb.append([m, b])

    # get median line
    mxb = np.array(mxb)
    if mxb.shape[0] == 0:
        print("No valid lines detected.")
    else:
        median_m = np.median(mxb[:, 0])
        median_b = np.median(mxb[:, 1])

    above = []
    below = []
    for pt in coords:
        x, y = pt
        y_on_line = median_m * x + median_b
        if y < y_on_line:  # image y-axis: 'above' line means numerically less
            above.append(pt)
        else:
            below.append(pt)

    above = np.array(above)
    below = np.array(below)
    if plot:
        for pt in coords:
            x, y = int(pt[0]), int(pt[1])
            y_on_line = median_m * x + median_b
            if y < y_on_line:
                label = "Above"
                color = (0, 255, 0)  # Green
            else:
                label = "Below"
                color = (0, 0, 255)  # Red
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.putText(
                image,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        # Draw Median line green
        height, width = image.shape[:2]
        x1, x2 = 0, width - 1
        y1 = int(median_m * x1 + median_b)
        y2 = int(median_m * x2 + median_b)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the lines on the original image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the result
        cv2.imshow('Lines Detected', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return above, below


def filter_close_points(coords, delta=10):
    # HELPER FUNCTION TO DELETE MULTIPLE DETECTIONS ON THE SAME OBJECT
    coords = np.array(coords)
    filtered = []
    used = np.zeros(len(coords), dtype=bool)

    for i, point in enumerate(coords):
        if used[i]:
            continue
        # Mark all points within delta as used
        dists = np.linalg.norm(coords - point, axis=1)
        close_idxs = np.where(dists < delta)[0]
        used[close_idxs] = True
        filtered.append(point)
    return np.array(filtered)


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


def parse_source_file(filename, width, height):
    points = []
    with open(filename, "r") as f:
        for line in f:
            # Remove whitespace and split by comma or space
            line = line.strip()
            # skip comments:
            if '#' in line:
                continue
            elif ',' in line:
                x_str, y_str = line.split(',')
            else:
                x_str, y_str = line.split()

            x_int = int(x_str)
            y_int = int(y_str)

            if x_int > width or x_int < 0 or y_int > height or y_int < 0:
                return [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            else:
                points.append([x_int, y_int])

    points = np.array(points, dtype=np.int32)
    if points.shape != (4, 2):
        return [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    return points


def export_vehicle_data(vehicle_data, filename):
    os.makedirs("csv_exports", exist_ok=True)
    filepath = os.path.join("csv_exports", filename)
    # CSV Export
    with open(filepath, mode="w", newline="") as csv_file:
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


if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", type=str, help="Relative path to video file")
    parser.add_argument("-d", "--distance", type=str, help="Distance between two delineators (in meters, default: 50m)")
    parser.add_argument("-m", "--model", type=str,
                        help="Optional: YOLO model used for vehicle detection (e.g., 'yolo12l.pt')")
    parser.add_argument("--plot", action="store_true", help="Enables plotting for debugging or visualization")
    parser.add_argument("--source_file", action="store_true",
                        help="Optional: Avoid detection, use custom source points in the text file")
    parser.add_argument("-c", "--calibration", type=str, help="Either type \"ransac\" or \"houghline\" for calibration")
    parser.add_argument("-s", "--section", type=str, help="Integer in which section to do analysis [0,1,2,3..]")

    args = parser.parse_args()

    videoPath = args.path if args.path else "example-videos/example.mp4"
    DISTANCE = args.distance if args.distance else 50.0  # default 50 meters
    DISTANCE = int(DISTANCE) if DISTANCE.isdigit() else 50
    modelType = args.model if args.model else "yolo12l.pt"
    plot = True if args.plot else False
    calibration = args.calibration if args.calibration else "ransac"
    s = args.section if args.section else "0"
    s = int(s) if s.isdigit() else 0

    video_info = sv.VideoInfo.from_video_path(videoPath)
    model = YOLO(modelType)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Get video infos
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    custom_source_flag = False
    sections = []
    source = np.array([])
    if args.source_file:
        custom_source = parse_source_file("custom_source_points.yaml", video_info.width, video_info.height)
        custom_source_flag = True
        if custom_source.any() == -1:
            custom_source_flag = False
        else:
            source = custom_source
    if not custom_source_flag:
        paired_delineators = get_coordinates(videoPath, calibration, plot)
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
        source = np.array([sections[s]["top_0"],  # A
                           sections[s]["top_1"],  # B
                           sections[s]["bottom_1"],  # C
                           sections[s]["bottom_0"]]  # D
                          , dtype="int32")

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
        if plot:
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

    base, _ = os.path.splitext(os.path.basename(videoPath))
    export_vehicle_data(vehicle_data, base + '.csv')
