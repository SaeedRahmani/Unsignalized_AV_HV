import argparse
from src.intersection.waymo.core import filter_all_unsignalized_intersections


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
        description="Filter intersection scenes from Waymo dataset")
    parser.add_argument("--version",
        default="v1.2.1", type=str, required=True, choices=["v1.2.1", "v1.3.0"],
        help="Version of Waymo motion dataset to load, choose from 'v1.2.1' and 'v1.3.0'")
    args = parser.parse_args()

    # main func
    filter_all_unsignalized_intersections(
        version=args.version, distance_threshold=45)

