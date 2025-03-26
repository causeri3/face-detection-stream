from utils.render import render_box, render_text

import logging
import cv2
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_payload(detected_objects):
    log_list = ['LABEL', 'CONFIDENCE']
    tags_list = []
    object_list = []

    for idx in range(len(detected_objects)):
        box = detected_objects[idx]
        label = str(box.data["class_name"].tolist()[0])
        confidence = "{:.2f}".format(box.confidence[0])

        log_list.append(label)
        log_list.append(confidence)

        object_list.append(label)
        tag = {
            "label": label,
            "score": confidence,
            "box": {
                "x": box.xyxy[0][0],
                "y": box.xyxy[0][1],
                "width": box.xyxy[0][3] - box.xyxy[0][1],
                "height": box.xyxy[0][2] - box.xyxy[0][0],
            },
            "tag_id": box.class_id
        }
        tags_list.append(tag)

    log_str_format = ' \n {:<40} {:<40}' * (len(detected_objects) + 1)
    logging.info(log_str_format.format(*log_list))

    dict = {
        "tags": tags_list,
        "objects": object_list
    }

    return json.dumps(dict, cls=NpEncoder)


def image_payload(detected_objects, image):
    """
    :param detected_objects:
    :param image: numpy array with three dimensions (height, width, channels)
    :return: images with bounding boxes drawn on, as bytes
    """
    log_list = ['LABEL', 'CONFIDENCE']
    output_image = image.copy()

    for idx in range(len(detected_objects)):
        box = detected_objects[idx]
        label = str(box.data["class_name"].tolist()[0])
        confidence = "{:.2f}".format(box.confidence[0])
        log_list.append(label)
        log_list.append(confidence)

        tag_text = label + ": " + confidence

        output_image = render_box(output_image, tuple(box.xyxy[0]))
        output_image = render_text(output_image, tag_text,
                                   (box.xyxy[0][0], box.xyxy[0][1]), normalised_scaling=0.5)

    success, encoded_image = cv2.imencode('.jpg', output_image)

    if not success:
        logging.error("""
        Could not encode image in order to convert to bytes {}. 
        """. format(type(output_image)))

    image_bytes_output = encoded_image.tobytes()

    log_str_format = ' \n {:<40} {:<40}' * (len(detected_objects) + 1)
    logging.info(log_str_format.format(*log_list))

    return image_bytes_output


