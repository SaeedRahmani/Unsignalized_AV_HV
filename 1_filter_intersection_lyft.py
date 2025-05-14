import argparse
from src import filter_all_unsignalized_intersections


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
        description="Filter intersection scenes from Lyft dataset")
    parser.add_argument("--type",
        default="sample", type=str, required=True, choices=["sample", "v1.3.0_training_20s", "validate"],
        help="Type of dataset to load, choose from 'sample', 'v1.3.0_training_20s', 'validate'")
    parser.add_argument("--id",
        default=0, type=int, required=True, choices=[0, 1],
        help="ID of dataset to load, 0 for 'WTgZ' and 1 for 'sGK1'")
    args = parser.parse_args()

    # main func
    filter_all_unsignalized_intersections(
        dataset_type=args.type, intersection_id=args.id)