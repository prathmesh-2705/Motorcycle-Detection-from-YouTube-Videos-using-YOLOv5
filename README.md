# Motorcycle Detection from YouTube Videos using YOLOv5

This project is a simple implementation of real-time motorcycle detection using YOLOv5, OpenCV, and YouTube video streaming with yt\_dlp. The idea is to pull a video directly from YouTube, process each frame, and detect motorcycles using a pre-trained deep learning model.

---

## Features

* Detects motorcycles in real-time
* Uses pre-trained YOLOv5s model
* Pulls video directly from YouTube (no downloading required)
* Displays the detection output using OpenCV

---

## Installation

Install the required packages using pip:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install yt-dlp
```

---

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/yourusername/motorcycle-detection-yolov5.git
cd motorcycle-detection-yolov5
```

2. Edit the script if needed and run:

```bash
python detect_motorcycles.py
```

Make sure you’re connected to the internet, as the script streams the video from YouTube and loads the YOLO model.

---

## Code Structure

* `detect_motorcycles.py` – main Python file that:

  * streams a YouTube video
  * processes frames using YOLOv5
  * displays real-time detection of motorcycles

---

## Known Issues

* Some videos may not stream properly depending on the format. If this happens, consider downloading the video manually.
* Detection is limited to motorcycles as labeled in the COCO dataset.

---

## To Do

* Add helmet detection
* Upgrade to YOLOv8
* Save output video or generate summary reports

---

## Credits

* YOLOv5 by Ultralytics
* yt\_dlp for YouTube streaming
* OpenCV for video processing

---

## License

This project is open-source and available under the MIT License.
