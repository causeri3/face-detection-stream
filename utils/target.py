"""
There are three states:
ALONE, BASIC_BITCH_STARE, and CLOSEUP_STARE.

ALONE:
IF there are no faces detected: The target point does a random walk across the screen. ALONE continues indefinitely until a face with an ID appears, at which point it switches to BASIC_BITCH_STARE or CLOSEUP_STARE.

BASIC_BITCH_STARE:
The target point follows the center in the upper third of a randomly chosen face ID. This state lasts for a random time range. If the time runs out or the face disappears, a new face is picked, and a new timer starts.

CLOSEUP_STARE:
Triggered immediately whenever a face larger than face_size_threshold of the screen appears. The target point locks onto a random chosen close-up face and keeps following it indefinitely. If the current close-up face disappears, a new close face is chosen randomly. It only exits CLOSEUP_STARE if no close face is detected.

"""

import time
import random
import cv2
import numpy as np

from utils.args import get_args
from utils.render import PINK

args, unknown = get_args()


class EyesTarget:
    def __init__(self):
        # "ALONE" or "BASIC_BITCH_STARE" or "CLOSEUP_STARE"
        self.state = "ALONE"

        # the face id we are currently staring at
        self.current_id = None
        self.valid_faces = []

        self.target_x = 0
        self.target_y = 0

        #_____________ Params to play with __________________ #
        self.stare_min_time = args.min_seconds #sec, int
        self.stare_max_time = args.max_seconds #sec, int
        self.face_size_threshold = args.perc_close  # % of screen area, float
        self.pause_min_time = 1  # Minimum pause time in seconds
        self.pause_max_time = 3  # Maximum pause time in seconds
        #______________________________________________________ #

        self.state_start_time = time.time()
        self.state_duration = 0
        self.last_move_time = time.time()


    def _reset_state(self,
                     new_state: str):
        """Helper to reset timings when changing states."""
        self.state = new_state
        self.state_start_time = time.time()
        if self.state == "BASIC_BITCH_STARE":
            self.state_duration = random.uniform(self.stare_min_time, self.stare_max_time)
        else:
            self.state_duration = 0
        self.current_id = self.pick_random_id()

    def pick_random_id(self) -> int | None:
        if not self.valid_faces:
            return None
        return random.choice(self.valid_faces)['id']

    def random_walk_with_pause(self,
                               width: float,
                               height: float,
                               last_x: float,
                               last_y: float) -> tuple[float, float]:
        """Simulate a random walk with pauses."""

        if last_x == 0 and last_y == 0:
            last_x = width / 2
            last_y = height / 2
        # move max 10% of width of screen / box
        move_range = int(round(width * 0.1))

        current_time = time.time()
        time_since_last_move = current_time - self.last_move_time

        # if outside of pause, update position
        if time_since_last_move > random.uniform(self.pause_min_time, self.pause_max_time):
            # Randomly choose a new position with small steps
            new_x = last_x + random.randint(-move_range, move_range)
            new_y = last_y + random.randint(-move_range, move_range)

            # force position on screen
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))

            self.last_move_time = current_time
        else:
            # if still in a pause
            new_x, new_y = last_x, last_y

        return new_x, new_y

    def update(self,
               faces:list[dict],
               image:np.ndarray) -> tuple[float, float]:
        """
        Updates the internal state given the detected faces and current screen shape.
        Returns (x, y) coordinates where the head should point.

        :param faces: list of dicts, each like:
            {
              'box': {'x': 32.79, 'y': 459.88, 'width': 411.75, 'height': 320.36},
              'confidence': '0.81',
              'id': 1 or None,
              'label': 'FACE',
              'label_int': 0
            }
        :param image: np.ndarray
        :return: (target_x, target_y) for head to look at
        """
        current_time = time.time()
        time_in_state = current_time - self.state_start_time

        screen_h, screen_w, _ = image.shape
        screen_area = screen_w * screen_h

        # filters out faces that actually have an ID, if deepsort track.is_confirmed()
        self.valid_faces = [f for f in faces if f['id'] is not None]

        ids_available = (len(self.valid_faces) > 0)

        close_faces = []
        for f in self.valid_faces:
            w = f['box']['width']
            h = f['box']['height']
            area = w * h
            if area >= self.face_size_threshold * screen_area:
                close_faces.append(f)

        # ______________________ react to close face _______________________ #
        react = (len(close_faces) > 0)
        if react:
            self.valid_faces = close_faces
            # If multiple large faces, see if current face is still among them
            current_big = any(f['id'] == self.current_id for f in close_faces)
            if not current_big:
                self.current_id = self.pick_random_id()
            if self.state != "CLOSEUP_STARE":
                self._reset_state("CLOSEUP_STARE")
        else:
            if self.state == "CLOSEUP_STARE":
                self._reset_state("BASIC_BITCH_STARE" if ids_available else "ALONE")

        # ______________________ chose states _______________________ #

        if not ids_available and not self.state == "ALONE":
            self._reset_state("ALONE")

        if self.state == "ALONE" and ids_available:
            self._reset_state("BASIC_BITCH_STARE")

        elif self.state == "BASIC_BITCH_STARE":
            still_see_current = any(f['id'] == self.current_id for f in self.valid_faces)
            if time_in_state < self.state_duration and still_see_current:
                pass
            else:
                # stare time is up OR we lost sight of current face, pick  new id at random
                self._reset_state("BASIC_BITCH_STARE")

        # ______________________ define new target spot _______________________ #
        if self.state == "ALONE":
            if int(time_in_state) % 2 == 0:
                # Every 2 seconds, pick a new random location using random walk
                self.target_x, self.target_y = self.random_walk_with_pause(screen_w, screen_h, self.target_x, self.target_y)



        elif self.state in ["BASIC_BITCH_STARE", "CLOSEUP_STARE"]:
            # Find the face with self.current_id, if it exists
            for f in self.valid_faces:
                if f['id'] == self.current_id:
                    # the center of that face's box
                    self.target_x = f['box']['x'] + f['box']['width'] / 2
                    self.target_y = f['box']['y'] + f['box']['height'] / 3
                    break

        # for debugging
        screen_logs = """{} Target ID: {} | Countdown or Time in State: {:.0f} | ids_available: {} | any close faces: {}""".format(
                            self.state,
                                    self.current_id,
                                    - (self.state_duration - time_in_state),
                                    ids_available,
                                    (len(close_faces) > 0)
                                    )
        cv2.putText(
            image,
            text=screen_logs,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color= PINK,
            thickness=2
        )

        return self.target_x, self.target_y, self.current_id, self.state

