import cv2
import numpy as np

_LINE_THICKNESS_SCALING = 500.0
_TEXT_THICKNESS_SCALING = 700.0
_TEXT_SCALING = 520.0
LIGHT_GREY = (220, 220, 220)
PINK = (255, 51, 255)


def render_box(img: np.ndarray,
               box: tuple[float, float, float, float],
               color: tuple[int, int, int] = LIGHT_GREY) -> np.ndarray:
    """
    Draws a box on the image with thickness scaled based on image size.
    :param img: image to draw on
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color. Default is white.
    :return: image with the rendered box.
    """
    x1, y1, x2, y2 = box
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_LINE_THICKNESS_SCALING * _LINE_THICKNESS_SCALING)
        )
    )
    thickness = max(1, thickness)
    img = cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness=thickness
    )
    return img


def get_text_size(img: np.ndarray,
                  text: str,
                  normalised_scaling: float = 1.0) -> tuple[int, int]:
    """
    Calculates the pixel dimensions (width, height) of the text based on image size.
    :param img: image reference, used to determine appropriate text scaling
    :param text: text to display
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: (width, height) - width and height of text box
    """
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)
        )
        * normalised_scaling
    )
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scaling, thickness)[0]


def render_text(img: np.ndarray,
                text: str,
                pos: tuple[float, float],
                color: tuple[int, int, int] = LIGHT_GREY,
                normalised_scaling: float = 1.0) -> np.ndarray:
    """
    Render a text into the image. Calculates scaling and thickness automatically.
    :param img: image to draw on.
    :param text: text to display.
    :param pos: (x, y) - upper left coordinates of render position.
    :param color: (b, g, r) - text color. Default white.
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: image with the rendered text.
    """
    x, y = pos
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)
        )
        * normalised_scaling
    )
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    size = get_text_size(img, text, normalised_scaling)
    cv2.putText(
        img,
        text,
        (int(x), int(y + size[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scaling,
        color,
        thickness=thickness,
    )
    return img

def draw_target_dot(
    img: np.ndarray,
    target_coords: tuple[float, float],
    color: tuple[int, int, int] = PINK) -> np.ndarray:
    cv2.circle(
        img,
        (int(round(target_coords[0])), int(round(target_coords[1]))),
        radius=10,
        color=color,
        thickness=-1  # -1 => filled
    )
    return img

def draw_boxes(
        img: np.ndarray,
        box: tuple[float, float, float, float],
        label: str) -> np.ndarray:

    img = render_box(img, box)
    img = render_text(img, label, (box[0], box[1]))

    return img
