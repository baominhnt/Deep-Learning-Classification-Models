import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from data.augmentation import get_transforms
from models import build_model


def predict_single_image(image_path, checkpoint_path, config_path):
    """
    Loads an individual image, dynamically builds the target model from configuration,
    restores saved weights, and predicts the target sports class.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Parse configuration maps and restore training checkpoint metadata
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing target checkpoint asset at: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    config["model"]["num_classes"] = checkpoint["num_classes"]

    # Reconstruct the neural network architecture dynamically via the factory
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Process the image asset matching your specific notebook pipeline
    img = Image.open(image_path).convert("RGB")

    # Extract the shared validation transform from your data module
    transform = get_transforms()["val"]
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Execute prediction forward pass
    with torch.no_grad():
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)[0]
        conf, preds = torch.max(probabilities, dim=0)

    predicted_class = class_names[preds.item()]
    confidence_score = conf.item() * 100

    print("\n[ Prediction Result ]")
    print(f"Target Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence_score:.2f}%")

    # 5. Display the image using your preferred layout style
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(
        f"Pred: {predicted_class} ({confidence_score:.2f}%)", fontsize=12, weight="bold"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference Interface")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to test image file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pth weight file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to target configuration yaml profile",
    )

    args = parser.parse_args()
    predict_single_image(args.image, args.checkpoint, args.config)
