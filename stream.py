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

