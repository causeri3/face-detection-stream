# Follow Faces

## Description
* **Face detection** implemented for camera stream with YOLOv8 trained on faces by [arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

* **Face tracking** implemented with deepsort realtime [Kalman Filter (movement prediction) + Hungarian Algorithm (assignment) + embedding model (using clip_RN50x16)]

  
* **Target Coordinates Functionality**:
Three states:
    * ALONE: If no faces are detected, the target point performs a random walk across the screen with occasional pauses.

    * BASIC_BITCH_STARE: If faces are detected, the target point locks onto one face by its ID and follows the center of its bounding box for a random duration.

    * CLOSEUP_STARE: If a face occupies more than x% (default 25%) of the screen area, it is always immediately targeted and stared at, regardless of the current state.

    States and target IDs are re-evaluated when durations expire or when visibility changes.


* **Smooth Hardware Movement**: 
* Hardware position moves continuously toward updated target coordinates in a separated thread.

## Dependencies
### Python
I used python 3.13  - you could peg it with `uv venv --python 3.13` but properly not necessary

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
* stare duration (max and min)
* close face threshold (% of the screen)
---
