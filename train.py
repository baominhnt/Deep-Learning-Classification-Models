import argparse
import yaml
import torch
import torch.nn as nn
from data.dataset import get_dataloaders
from models import build_model
from engine import train_model, evaluate_final_test_set
from utils.helpers import save_checkpoint, plot_performance_curves


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Architecture Deep Learning Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to targeted model YAML file config",
    )
    args = parser.parse_args()

    # Parse target YAML profile parameters
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("--- Step 1: Loading Pipeline & Datasets ---")
    dataloaders, class_names, num_classes, dataset_sizes = get_dataloaders(
        config["data"]
    )

    print("\n--- Step 2: Assembling Model Infrastructure ---")
    config["model"]["num_classes"] = num_classes
    model = build_model(config["model"])

    # Define optimization pieces that engine.py expects
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-6
    )

    print("\n--- Step 3: Launching Optimization Engine ---")
    # MATCH INDICES: Pass the exact inputs your custom engine.py expects!
    trained_model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["train"]["device"],
        num_epochs=config["train"]["epochs"],
    )

    # ADDED: STEP 3.5 - RUN FINAL TEST SET EVALUATION
    print("\n--- Step 3.5: Evaluating Performance on Unseen Test Partition ---")
    test_loss, test_acc = evaluate_final_test_set(
        model=trained_model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=config["train"]["device"],
    )

    # Append test deliverables to your history object for metadata logging
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc

    print("\n--- Step 4: Exporting Run Deliverables ---")
    # Dynamic weight naming setup
    config["train"]["model_save_name"] = config["model"]["name"]
    save_checkpoint(trained_model, history, config["train"], num_classes, class_names)
    plot_performance_curves(history, config["train"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
