# Follow Faces

## Description
* **Face detection** implemented for camera stream with YOLOv8 trained on faces by [arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

* **Face tracking** 
  * implemented with deepsort realtime [Kalman Filter (movement prediction) + Hungarian Algorithm (assignment) + embedding model (using clip_RN50x16 or torchreid)]

    * **Target Coordinates Functionality**:
    Three states:
        * ALONE: If no faces are detected, the target point performs a random walk across the screen with occasional pauses.

        * BASIC_BITCH_STARE: If faces are detected, the target point locks onto one face by its ID and follows the center of its bounding box for a random duration.

        * CLOSEUP_STARE: If a face occupies more than x% (default 25%) of the screen area, it is always immediately targeted and stared at, regardless of the current state.

        States and target IDs are re-evaluated when duration expires or when visibility changes.

* **No Tracking**

    Alternative approach to tracking (fallback if tracking performance and or compute resources are not be sufficient.)

    Two states:

    * ALONE : same as above
    * STARE: If boxes are detected, one is selected as the target, alternating the biggest or a random boy. For 20 seconds, teh chosen target is the box currently closest to the last target point.

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

**Note**: 

If you want to use torchreid, you will have to in deepsort realtime source coe and change in `deep_sort_realtime/embedder/embedder_pytorch.py`:
Line 163

`tochreid.utils import FeatureExtractor`

to 

`tochreid.reid.utils import FeatureExtractor`

## Run
* with uv
 `uv run main.py`

* standard
 `python main.py`

### Args
See all arguments:
* with uv `uv run main.py --help`
* standard `python main.py --help`


You can 
* choose between multiple camera devices
* choose whether the object detection is streaming as video or just in the logs - both ways you get a json payload out with the detections
* set a confidence threshold
* set an IoU threshold
* stare duration (max and min)
* close face threshold (% of the screen)
* choose between tracking with deepsort or alternative approach without tracking
---
