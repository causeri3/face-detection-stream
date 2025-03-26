
from utils.video import Stream
from utils.args import get_args

import logging

args, unknown = get_args()
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)



if __name__ == "__main__":

    if args.cam_device_number:
        logging.info("""You chose camera device no: {}""".format(args.cam_device_number))
        device = [args.cam_device_number]
    else:
        device = None
    stream = Stream(
            see_detection=args.see_detection,
            available_devices = device)
    stream.draw_boxes()



# from huggingface_hub import hf_hub_download
# from ultralytics import YOLO
# from supervision import Detections
# from PIL import Image
# model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# model = YOLO(model_path)
# image_path = "/Users/vanessacausemann/Downloads/personal/Irland mit Mami/IMG_8265.JPG"
# #frame = cv2.imread(image_path)
# #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
# pil_image = Image.open(image_path)
# output = model(pil_image)
# results = Detections.from_ultralytics(output[0])
# x1, y1, x2, y2 = results.xyxy[0]
#

