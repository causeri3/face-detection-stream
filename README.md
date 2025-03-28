# Stream Face detection

## Description
Face detection implemented for camera stream with YOLOv8 trained by [arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

## Dependencies
### Python
I used python 3.13

you could peg it with 
`uv venv --python 3.13`
but properly not necessary

### Python packages 
You can install them via

`pip install -r requirements.txt`

Or even better if you use [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1), run in the directory of this repo:
```sh
uv venv
uv pip install -r requirements.txt
```
## Run
* with uv
 `uv run stream.py`

* standard
 `python stream.py`

### Args
See all arguments:
* with uv `uv run stream.py --help`
* standard `python stream.py --help`


You can 
* choose between multiple camera devices
* choose whether the object detection is streaming as video or just in the logs - both ways you get a json payload out with the detections
* set a confidence threshold
* set an IoU threshold
---
