import copy
import torch
from tqdm import tqdm


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
):
    """
    Core training and validation execution engine.
    """
    model = model.to(device)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize local history tracking
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # tqdm progress bar loop
            for inputs, labels in tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Batch", leave=False
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

                # Track checkpoints during validation phase
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "best_model_checkpoint.pth")
                    print(f"=> Saved new best model checkpoint (Acc: {best_acc:.4f}).")

    print(f"\nTraining complete! Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_final_test_set(model, dataloader, criterion, device):
    """
    Final evaluation engine on the unseen testing partition.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

    test_loss = running_loss / total_samples
    test_acc = running_corrects / total_samples

    print(f"\nTotal Test Images: {total_samples}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f} ({running_corrects}/{total_samples})")

    return test_loss, test_acc
