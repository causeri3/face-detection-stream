from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-cam',
                        '--cam-device-number',
                        required=False,
                        default=None,
                        type=int,
                        help='Overwrite camera device number (integer)')
    parser.add_argument('-sd',
                        '--see-detection',
                        required=False,
                        action='store_false',
                        help="""See object detection in streamed video output.
                        If set to False, you only get the json payload (boolean)""")
    parser.add_argument('-conf',
                        '--confidence-threshold',
                        required=False,
                        default=0.25,
                        type=float,
                        help="""Confidence threshold for detected object (float between 0 and 1)""")
    parser.add_argument('-iou',
                        '--iou-threshold',
                        required=False,
                        default=0.05,
                        type=float,
                        help="""Threshold for IoU - Intersection over Union (float between 0 and 1)""")

    # to not get into trouble with uvicorn args
    args = parser.parse_known_args()
    return args