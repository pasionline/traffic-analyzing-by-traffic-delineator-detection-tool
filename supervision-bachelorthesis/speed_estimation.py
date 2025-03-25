import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv



if __name__ == "__main__":

    videoPath = "highway_test.mp4"

    video_info = sv.VideoInfo.from_video_path(videoPath)
    model = YOLO("yolo11n.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Get video infos
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    # Lines:
    START1 = sv.Point(382, 456)
    END1 = sv.Point(846, 435)
    START2 = sv.Point(1742, 611)
    END2 = sv.Point(642, 711)

    line1 = sv.LineZone(start=START1, end=END1)
    line2 = sv.LineZone(start=START2, end=END2)

    line_zone_annotator = sv.LineZoneAnnotator(text_scale=2, text_thickness=4,display_out_count=False)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)



    frame_generator = sv.get_video_frames_generator(videoPath)

    # Process Video:
    for frame in frame_generator:
        result = model(frame, verbose=False)[0]
        # Object detection
        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections=detections)

        trigger1 = line1.trigger(detections)
        trigger2 = line2.trigger(detections)

        enteredVehicle = trigger1[0]
        leavingVehicle = trigger2[0]



        # Label each vehicle:
        labels = [
            f"#{tracker_id} {result.names[class_id]} {confidence:0.2f}"
            for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]
        annotated_frame = frame.copy()
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line1)
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line2)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)



        #Plotting:
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()