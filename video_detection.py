from ultralytics import YOLO
from collections import defaultdict
import cv2

def initialize_video_capture(video_path):
    """Initialize video capture object"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    return cap

def get_video_properties(cap):
    """Get video width, height, and fps"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, fps

def initialize_video_writer(output_path, fourcc, fps, frame_size):
    """Initialize video writer object"""
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def process_frame(model, frame, counts, frame_count, frame_rate_divider):
    """Process video frame and perform object detection"""
    if frame_count % frame_rate_divider != 0:
        return frame, counts, False

    results = model(frame)[0]

    for box in results.boxes:
        class_id = model.names[box.cls[0].item()]
        counts[class_id] += 1

        # Draw detection results on the frame
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[box.cls[0].item()]}: {box.conf[0]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, counts, True

def main():
    model = YOLO("yolov10x.pt")
    video_path = "input_video.mp4"
    output_path = "output_video.mp4"

    cap = initialize_video_capture(video_path)
    width, height, fps = get_video_properties(cap)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = initialize_video_writer(output_path, fourcc, fps, (width, height))

    frame_rate_divider = 1  # Adjust this value as needed
    frame_count = 0

    counts = defaultdict(int)
    object_str = ""
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, counts, processed = process_frame(model, frame, counts, frame_count, frame_rate_divider)

        if processed:
            key = f"({index}): "
            index += 1
            object_str += ". " + key
            for class_id, count in counts.items():
                object_str += f"{count} {class_id},"

            counts = defaultdict(int)

        out.write(frame)
        frame_count += 1

    object_str = object_str.strip(',').strip('.')
    print("result:", object_str)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


