import fiftyone as fo
import fiftyone.zoo as foz

classes = [
    "Person",
    "Dog",
    "Cat",
    "Horse",
    "Cattle",
    "Sheep",
    "Bird",
    "Elephant"
]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=500,
    # Change dataset_dir to download_dir
    download_dir=r"F:\Maharishi_assignment\datasets\open_images"
)

# Optional: If you want to see the images in the App after downloading
session = fo.launch_app(dataset)
session.wait()