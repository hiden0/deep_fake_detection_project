import json
import csv


def json_to_csv(json_file, csv_file):
    """
    Converts a JSON file to a CSV file.

    Args:
      json_file: Path to the JSON file.
      csv_file: Path to the output CSV file.
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    with open(csv_file, "w", newline="") as csvfile:
        fieldnames = ["filename", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for filename, video_data in data.items():
            writer.writerow({"filename": filename, "label": video_data["label"]})


# Example usage:
json_file = (
    "/srv/nvme/javber/deep_fake_detection_sample/train_sample_videos/metadata.json"
)
csv_file = (
    "/srv/nvme/javber/deep_fake_detection_sample/train_sample_videos/metadata.csv"
)

json_to_csv(json_file, csv_file)
