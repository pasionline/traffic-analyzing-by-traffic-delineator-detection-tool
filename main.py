import argparse

def parse_arguments() -> arg.parse.Namespace:
    parser = argparse.ArgumentParser(
        description='Bachelorarbeit Supervision'
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    pass