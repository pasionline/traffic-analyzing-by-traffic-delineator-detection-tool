# Traffic Analyzing Tool

**Description**  
This tool makes uses fine-tuned YOLOv12x [YOLOv12_traffic-delineator](https://huggingface.co/maco018/YOLOv12_traffic-delineator) model which detects traffic delineators and calculates the speed, acceleration and the average distance to other vehicles and saves those data to CSV.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Installation

### Prerequisites
- (Optional) If you want to run Torch with CUDA, make sure to get the correct version from: https://pytorch.org/get-started/locally/
- Python 3.12.9
### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/pasionline/traffic-analyzing-by-traffic-delineator-detection-tool.git
   ```
2. Navigate into the directory:
   ```bash
   cd traffic-analyzing-by-traffic-delineator-detection-tool
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
## Usage
For simply analyzing an existing video with a delineator-gap distance of 50 meters, use this command:
```bash
python analyze_video.py -p path/to/the/video.mp4
```

Run an analysis with a custom distance:

```bash
python analyze_video.py -p path/to/the/video.mp4 -d 100
```

### Optional:

Switch the YOLO model which detects vehicles (default model is yolo12l.pt:

```bash
python analyze_video.py -p path/to/the/video.mp4 -m yolo12n.pt
```

Enable plotting/visualization: 

```bash
python analyze_video.py -p path/to/the/video.mp4 --plot
```

## License

   
