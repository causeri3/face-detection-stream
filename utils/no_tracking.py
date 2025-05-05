"""
There are two states:
ALONE and STARE.

ALONE:
If there are no boxes detected: The target point performs a random walk across the screen, taking small steps (5% of the screen width) with random pauses between 1–3 seconds.
This state continues indefinitely until at least one box appears, at which point it switches to STARE.

STARE:
Triggered when one or more boxes are detected. The target point is set to the center-top-third of the selected box. A target box is selected by:
- choose a box every 20 sec - alternating biggest boy - random box
- while the 20 sec the target tracks the currently closest box to the last target position.
"""

import time
import random
import math
import numpy as np
import cv2

from utils.render import PINK

class SimpleTargetSelector:
    def __init__(self):
        self.target_x = None
        self.target_y = None
        self.last_switch_time = time.time()
        self.use_biggest_box = True
        self.boxes = []
        # state can be ALONE or STARE
        self.state = 'STARE'
        self.pause_min_time = 1  # minimum pause time in seconds
        self.pause_max_time = 3  # maximum pause time in seconds
        self.last_move_time = time.time()
        self.stare_duration = 20 # sec
        self.random_walk_step_size = 0.05 # move step size 5% of screen width


    def random_walk_with_pause(self,
                               width: float,
                               height: float,
                               last_x: float,
                               last_y: float) -> tuple[float, float]:
        """Simulate a random walk with pauses."""

        # move max 10% of width of screen / box
        move_range = int(round(width * self.random_walk_step_size))

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

    @staticmethod
    def get_target_coords(box: dict):
        x = box['box']['x'] + box['box']['width'] / 2
        y = box['box']['y'] + box['box']['height'] / 3
        return x, y

    def dist(self, box: dict):
        x, y = self.get_target_coords(box)
        return math.hypot(x - self.target_x, y - self.target_y)

    def choose_box(self):

        now = time.time()
        if now - self.last_switch_time >= self.stare_duration:
            self.last_switch_time = now
            self.use_biggest_box = not self.use_biggest_box

            if self.use_biggest_box:
                selected = max(self.boxes, key=lambda f: f['box']['width'] * f['box']['height'])
            else:
                selected = random.choice(self.boxes)
        else:
            selected = min(self.boxes, key=self.dist)

        self.target_x, self.target_y = self.get_target_coords(selected)


    def update_target(self,
                      boxes: list[dict],
                      image:np.ndarray):

        if not boxes:
            screen_h, screen_w, _ = image.shape
            if not self.state == 'ALONE':
                self.last_switch_time = time.time() - self.stare_duration
                self.state = 'ALONE'

            # start in the middle of the screen
            if self.target_x is None or self.target_y is None:

                self.target_x = screen_w / 2
                self.target_y = screen_h / 2

            self.target_x, self.target_y = self.random_walk_with_pause(
                width=screen_w,
                height=screen_h,
                last_x=self.target_x,
                last_y=self.target_y
            )
        else:
            self.state = 'STARE'
            self.boxes = boxes
            self.choose_box()

        # for debugging
        screen_logs = """ State {} |  Biggest Box {} | Countdown or Time in State: {:.0f}""".format(
                            self.state,
                                    self.use_biggest_box,
                                    - (self.stare_duration - (time.time() - self.last_switch_time))
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

        return self.target_x, self.target_y, self.state


