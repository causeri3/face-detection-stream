"""
There are two states:
STARE and BORED
They both take a random amount of time in between a max and min chosen range. Once time is up, they change.
Except, if there are no face ids, it will always be in BORED state.

In BORED:
the target point does a random walk with the maximum distance of 2*10% of the screen width
In STARE:
If no target id chosen or chosen one is not available, chooses one at random. Target point follows center of bounding box

Little extra:
Once a face comes closer than 25% of the screen, it gets stared at, which ever state we are in.
"""

import time
import random
import cv2
import numpy as np

from utils.render import PINK


class EyesTarget:
    def __init__(self):
        # "BORED" or "STARE"
        self.state = "BORED"

        # the face id we are currently staring at
        self.current_id = None

        self.target_x = 0
        self.target_y = 0

        #_____________ Params to play with __________________ #
        self.bored_min_time = 5
        self.bored_max_time = 15
        self.stare_min_time = 20
        self.stare_max_time = 40
        self.face_size_threshold = 0.25  # % of screen area
        self.pause_min_time = 1  # Minimum pause time in seconds
        self.pause_max_time = 3  # Maximum pause time in seconds
        #______________________________________________________ #

        self.state_start_time = time.time()
        # duration of state
        self.state_duration = random.uniform(self.bored_min_time, self.bored_max_time)
        self.last_move_time = time.time()


    def _reset_state(self, new_state: str):
        """Helper to reset timings when changing states."""
        self.state = new_state
        self.state_start_time = time.time()
        if new_state == "BORED":
            self.state_duration = random.uniform(self.bored_min_time, self.bored_max_time)
        else:  # STARE
            self.state_duration = random.uniform(self.stare_min_time, self.stare_max_time)


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
        valid_faces = [f for f in faces if f['id'] is not None]

        ids_available = (len(valid_faces) > 0)

        close_faces = []
        for f in valid_faces:
            w = f['box']['width']
            h = f['box']['height']
            area = w * h
            if area >= self.face_size_threshold * screen_area:
                close_faces.append(f)

        # ______________________ react to close face _______________________ #
        react = (len(close_faces) > 0)
        if react:
            # If multiple large faces, see if current face is still among them
            current_big = any(f['id'] == self.current_id for f in close_faces)
            if not current_big:
                chosen_face = random.choice(close_faces)
                self.current_id = chosen_face['id']
            # switch to STARE if not already
            if self.state != "STARE":
                self._reset_state("STARE")

        # ______________________ chose states _______________________ #
        if self.state == "BORED":
            if time_in_state < self.state_duration:
                pass
            else:
                # BORED time is up
                if ids_available:
                    chosen_face = random.choice(valid_faces)
                    self.current_id = chosen_face['id']
                    self._reset_state("STARE")
                else:
                    # No faces -> remain bored (reset bored timer)
                    self._reset_state("BORED")

        elif self.state == "STARE":
            still_see_current = any(f['id'] == self.current_id for f in valid_faces)
            if time_in_state < self.state_duration and still_see_current:
                pass
            else:
                # STARE time is up OR we lost sight of current face
                self._reset_state("BORED")

        # ______________________ define new target spot _______________________ #
        if self.state == "BORED":
            if int(time_in_state) % 2 == 0:
                # every 2 seconds, pick a new random location
                # self.target_x = random.uniform(0, screen_w)
                # self.target_y = random.uniform(0, screen_h)

                # Every 2 seconds, pick a new random location using random walk
                self.target_x, self.target_y = self.random_walk_with_pause(screen_w, screen_h, self.target_x, self.target_y)


        elif self.state == "STARE":
            # Find the face with self.current_id, if it exists
            for f in valid_faces:
                if f['id'] == self.current_id:
                    # the center of that face's box
                    self.target_x = f['box']['x'] + f['box']['width'] / 2
                    self.target_y = f['box']['y'] + f['box']['height'] / 2
                    break

        # for debugging
        screen_logs = """{} Target ID: {} Countdown: {:.0f} ids_available: {}, any close faces: {}""".format(
                            self.state,
                                    self.current_id,
                                    self.state_duration - time_in_state,
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

        # cv2.putText(
        #     image,
        #     text="{}, {}".format(self.target_x, self.target_y),
        #     org=(10, 100),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=1.0,
        #     color= PINK,
        #     thickness=2
        # )

        return self.target_x, self.target_y
