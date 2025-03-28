from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-cam',
                        '--cam-device-number',
                        required=False,
                        default=None,
                        type=int,
                        help='Overwrite camera device number with an integer')
    parser.add_argument('-sd',
                        '--see-detection',
                        required=False,
                        action='store_false',
                        help="""See object detection in streamed video output.
                        If set to False, you only get the json payload""")

    # to not get into trouble with uvicorn args
    args = parser.parse_known_args()
    return args