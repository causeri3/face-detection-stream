from utils.render import render_box, render_text

import logging
import cv2
import numpy as np
from supervision.detection.core import Detections

def json_payload(detected_objects: Detections) -> dict:
    bbs_list = []
    tags_list = []

    for idx in range(len(detected_objects)):
        box = detected_objects[idx]
        label = str(box.data["class_name"].tolist()[0])
        confidence = "{:.2f}".format(box.confidence[0])

        x = box.xyxy[0][0]
        y = box.xyxy[0][1]
        height = box.xyxy[0][2] - box.xyxy[0][0]
        width = box.xyxy[0][3] - box.xyxy[0][1]
        detected_class_int = box.class_id[0]
        tag = {
            "label": label,
            "score": confidence,
            "box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "tag_id": detected_class_int
        }

        bbs_list.append([
            [x, y, width, height],
            confidence,
            detected_class_int
        ])
        tags_list.append(tag)

    bbs_dict = {
        "tags": tags_list,
        "bbs": bbs_list
    }
    logging.debug(bbs_dict["tags"])
    return bbs_dict


def image_payload(detected_objects: Detections,
                  image: np.ndarray,
                  track_ids: list):
    """
    :param detected_objects:
    :param image: numpy array with three dimensions (height, width, channels)
    :param track_ids: list with all tracked ids

    :return: images with bounding boxes drawn on, as bytes
    """
    log_list = ['LABEL', 'CONFIDENCE', "ID"]
    output_image = image.copy()

    for idx in range(len(detected_objects)):
        box = detected_objects[idx]
        label = str(box.data["class_name"].tolist()[0])
        confidence = "{:.2f}".format(box.confidence[0])
        track_id = track_ids[idx]
        log_list.append(label)
        log_list.append(confidence)
        log_list.append(track_id)

        tag_text = label + ": " + confidence + " ID: " + track_id

        output_image = render_box(output_image, tuple(box.xyxy[0]))
        output_image = render_text(output_image, tag_text,
                                   (box.xyxy[0][0], box.xyxy[0][1]), normalised_scaling=0.5)

    success, encoded_image = cv2.imencode('.jpg', output_image)

    if not success:
        logging.error("""
        Could not encode image in order to convert to bytes {}. 
        """. format(type(output_image)))

    image_bytes_output = encoded_image.tobytes()

    log_str_format = ' \n {:<40} {:<40} {:<40}' * (len(detected_objects) + 1)
    logging.info(log_str_format.format(*log_list))

    return image_bytes_output


