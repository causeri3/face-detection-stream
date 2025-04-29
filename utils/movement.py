import logging
import time
from utils.payloads import get_target_box_size

_FREQUENCY = .01 #sec
_ALLOWED_SLACK = .005 #sec

def hardware_movement_loop(target_lock, dict_payload):
    current_position = [0, 0]
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > _FREQUENCY + _ALLOWED_SLACK:

            logging.warning("""
            Movement delayed by {:.4f} seconds.
            If this escalates, another option would be to try multiprocessing instead of threading.
            That should keeps this loop timing stable, but increase overall computing costs (overhead).
            """.format(dt - _FREQUENCY))

        # get values from main thread
        with target_lock:
            # numpy floats64
            x, y = dict_payload['target_coordinates_xy']
            logging.debug(f"x: {x}, y: {y}")
            # string
            state = dict_payload['state']
            logging.debug(f"state: {state}")
            if dict_payload['target_id']:
                # basic bitch floats
                width, height = get_target_box_size(dict_payload)
                logging.debug(f"width: {width}, height: {height}")

        # TODO: Do your thing, Mateo
        speed = 0.05
        current_position[0] += (x - current_position[0]) * speed
        current_position[1] += (y - current_position[1]) * speed

        # Send to your hardware
        _send_to_hardware(current_position)

        time.sleep(_FREQUENCY)

def _send_to_hardware(position):
    # TODO: Keinen plan wie deine connection war
    logging.debug(f"Move to {position}")