import torch
import cv2
import yt_dlp
import numpy as np

def get_video_stream(url):
    ydl_opts = {'quiet': True, 'format': 'best'}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']
    
    return cv2.VideoCapture(video_url)

def detect_motorcycles(frame, model):
    results = model(frame)
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if model.names[int(cls)] == 'motorcycle':
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Motorcycle {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    youtube_url = "https://youtu.be/2_1xD9ipDaA"
    cap = get_video_stream(youtube_url)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = detect_motorcycles(frame, model)

        cv2.imshow('Motorcycle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
