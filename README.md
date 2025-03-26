# Stream Face detection

## Description
Face detection implemented for camera stream with YOLOv8 trained by [arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

## Dependencies
### Python
I used
python 3.13
you could peg it with 
`uv venv --python 3.13`
but properly not necessary

### Python packages 
You can install them via

`pip install -r requirements.txt`

or even better if you use uv:
```sh
uv venv
uv pip install -r requirements.txt
```
## Run
 `python stream.py --help`

### Args
See all arguments : `python stream.py --help`

You can 
* choose between multiple camera devices
* choose whether the object detection is in your video or just in the logs
* choose between an image or json pyload

---
