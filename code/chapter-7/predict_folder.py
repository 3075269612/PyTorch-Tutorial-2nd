import argparse
import csv
import os

import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms


BASEDIR = os.path.dirname(os.path.abspath(__file__))
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(num_classes=2):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transform():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


def collect_images(folder_path):
    image_paths = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in VALID_EXTENSIONS:
            image_paths.append(file_path)
    return image_paths


def predict_folder(folder_path, weight_path, output_csv):
    device = torch.device("cpu")
    class_names = ["ants", "bees"]

    if not os.path.isdir(folder_path):
        raise NotADirectoryError("Folder not found: {}".format(os.path.abspath(folder_path)))

    if not os.path.isfile(weight_path):
        raise FileNotFoundError("Weight file not found: {}".format(os.path.abspath(weight_path)))

    image_paths = collect_images(folder_path)
    if not image_paths:
        raise FileNotFoundError("No images found in folder: {}".format(os.path.abspath(folder_path)))

    model = build_model(num_classes=len(class_names))
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = build_transform()
    rows = []
    class_count = {name: 0 for name in class_names}

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_index = torch.argmax(probabilities).item()
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index].item()
            class_count[predicted_class] += 1
            rows.append([
                os.path.abspath(image_path),
                predicted_class,
                "{:.6f}".format(confidence),
                "{:.6f}".format(probabilities[0].item()),
                "{:.6f}".format(probabilities[1].item()),
            ])

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "predicted_class", "confidence", "ants_prob", "bees_prob"])
        writer.writerows(rows)

    print("Processed images:", len(rows))
    for class_name in class_names:
        print("{}: {}".format(class_name, class_count[class_name]))
    print("Prediction csv saved to:", os.path.abspath(output_csv))


def parse_args():
    parser = argparse.ArgumentParser(description="Predict all images in a folder with the finetuned ResNet18 model.")
    parser.add_argument("folder_path", help="Path to the folder containing images.")
    parser.add_argument(
        "--weight-path",
        default=os.path.join(BASEDIR, "my_finetuned_model.pth"),
        help="Path to the saved state_dict file.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to save the csv prediction results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(args.folder_path, "batch_predictions.csv")
    predict_folder(args.folder_path, args.weight_path, output_csv)