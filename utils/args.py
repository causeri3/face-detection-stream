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
                        If argument in command, you only get the json payload (boolean)""")

    parser.add_argument('-nt',
                        '--no-tracking',
                        required=False,
                        action='store_false',
                        help="""Use tracking to get target coordinates.
                        If argument in command, you will not use any tracking""")

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
    parser.add_argument('-max',
                        '--max-seconds',
                        required=False,
                        default=40,
                        type=int,
                        help="""Maximum amount of seconds to follow a face in BASIC_BITCH_STARE state""")
    parser.add_argument('-min',
                        '--min-seconds',
                        required=False,
                        default=20,
                        type=int,
                        help="""Minimum amount of seconds to follow a face in BASIC_BITCH_STARE state""")
    parser.add_argument('-close',
                        '--perc-close',
                        required=False,
                        default=.25,
                        type=float,
                        help="""% of screen which are considered a close face, if the bounding box fills it""")

    # to not get into trouble with uvicorn args
    args = parser.parse_known_args()
    return args