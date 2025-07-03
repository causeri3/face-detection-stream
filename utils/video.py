import cv2
import logging
import numpy as np
import time
import _thread

from utils.predict import Predict

class Stream:
    def __init__(self,
                 see_detection: bool = True,
                 available_devices: list | None = None):

        self.predict_class = Predict()

        self.frame = None
        self.available_devices = available_devices
        self.see_detection = see_detection

    @staticmethod
    def return_camera_indexes():
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
                logging.debug("Found device under number {}.".format(index))
            index += 1
            i -= 1
        logging.info("Available devices found: {}".format(arr))
        return arr

    @staticmethod
    def choose_device(device_numbers: list):
        if len(device_numbers) < 2:
            return device_numbers[0]
        # default for my preferred set-up (no deeper meaning)
        elif len(device_numbers) == 3:
            return device_numbers[1]
        else:
            return device_numbers[-1]

    def predict_n_stream(self):

        if self.see_detection:
            combined_image_bytes, json_payload = self.predict_class.predict(
                self.frame,
                return_image=True)

            combined_image_array = np.frombuffer(combined_image_bytes, dtype=np.uint8)
            combined_img = cv2.imdecode(combined_image_array, flags=1)
            return combined_img, json_payload

        else:
            json_payload = self.predict_class.predict(
                self.frame,
                return_image=False)
            return json_payload


    def draw_boxes(self,
                   target_lock:_thread.lock=None,
                   dict_payload:dict = None):

        if not self.available_devices:
            self.available_devices = self.return_camera_indexes()

        device_numbers = self.choose_device(self.available_devices)
        # Initialize the webcam
        cap = cv2.VideoCapture(device_numbers)
        window_name = "Your camera, device no: {}".format(device_numbers)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while cap.isOpened():
            start_time = time.time()

            # Read frame from the video
            ret, self.frame = cap.read()

            if not ret:
                logging.warning("Failed to read frame. Retry...")
                continue

            if self.see_detection:
                image, json_payload = self.predict_n_stream()
                cv2.imshow(window_name, image)
                # Press key q to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                json_payload = self.predict_n_stream()
                logging.info(json_payload)

            if target_lock:
                with target_lock:
                    dict_payload.clear()
                    dict_payload.update(json_payload)

            logging.info("Processing one frame took {:.2f} sec".format(time.time() - start_time))

