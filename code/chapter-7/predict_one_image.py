import argparse
import os

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torchvision.models as models
import torchvision.transforms as transforms


BASEDIR = os.path.dirname(os.path.abspath(__file__))


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


def save_visualization(image, predicted_class, confidence, output_path):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    text = "{} ({:.2%})".format(predicted_class, confidence)
    draw.rectangle((8, 8, 260, 40), fill=(0, 0, 0))
    draw.text((12, 14), text, fill=(255, 255, 255))
    annotated_image.save(output_path)


def predict_image(image_path, weight_path, output_path=None):
    device = torch.device("cpu")
    class_names = ["ants", "bees"]

    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image not found: {}".format(os.path.abspath(image_path)))

    if not os.path.isfile(weight_path):
        raise FileNotFoundError("Weight file not found: {}".format(os.path.abspath(weight_path)))

    model = build_model(num_classes=len(class_names))
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    print("Image:", os.path.abspath(image_path))
    print("Predicted class:", class_names[predicted_index])
    print("Confidence: {:.2%}".format(probabilities[predicted_index].item()))
    print("All probabilities:")
    for index, class_name in enumerate(class_names):
        print("  {}: {:.2%}".format(class_name, probabilities[index].item()))

    if output_path is None:
        image_name, image_ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(BASEDIR, "{}_prediction{}".format(image_name, image_ext))
    save_visualization(image, class_names[predicted_index], probabilities[predicted_index].item(), output_path)
    print("Visualization saved to:", os.path.abspath(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a single image with the finetuned ResNet18 model.")
    parser.add_argument(
        "image_path",
        help="Path to the image to classify.",
    )
    parser.add_argument(
        "--weight-path",
        default=os.path.join(BASEDIR, "my_finetuned_model.pth"),
        help="Path to the saved state_dict file.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to save the annotated prediction image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_image(args.image_path, args.weight_path, args.output_path)