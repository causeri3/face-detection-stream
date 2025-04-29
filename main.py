from utils.video import Stream
from utils.args import get_args
from utils.movement import hardware_movement_loop

import threading
import logging

args, unknown = get_args()
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


target_lock = threading.Lock() # to prevent race conditions
dict_payload = {'objects': [], 'bbs': [], 'target_coordinates_xy': (.0, .0), 'target_id': None, 'state': 'NULL'}


if __name__ == "__main__":

    if args.cam_device_number:
        logging.info("""You chose camera device no: {}""".format(args.cam_device_number))
        device = [args.cam_device_number]
    else:
        device = None
    stream = Stream(
            see_detection=args.see_detection,
            available_devices = device)
    #stream.draw_boxes()

    movement_thread = threading.Thread(target=hardware_movement_loop, args=(target_lock, dict_payload))
    movement_thread.start()

    # OpenCV MUST run in main thread
    stream.draw_boxes(target_lock, dict_payload)

    movement_thread.join()


"""
Ye,ye, CPU heavy - use multiprocessing, I know. 
However, the hardware control is mainly I/O stuff. 
And I speculate that most CPU heavy stuff (yolov8 and the embeddings in deepsort, camera) are done under hood through 
numpy, PyTorch and maybe tensorflow which can circumvent the GIL via C/C++. So threading should have less overhead. 
I made the experience that it often runs faster. 
Haven’t tested it here yet, though. So if shit is laggy, feel free to play with multiprocessing.
"""

