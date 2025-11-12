import torch
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """Implements the Focal Loss as a custom PyTorch module.

    Focal Loss is particularly useful for addressing class imbalance in
    classification tasks. It modifies the standard cross-entropy loss
    by adding a scaling factor that decreases as confidence in the
    correct prediction increases, helping to focus more on hard-to-classify
    samples.

    Attributes:
        gamma (float): Focusing parameter that controls the strength of the
            modulation applied to the cross-entropy loss. Higher values of
            gamma lead to greater emphasis on hard-to-classify samples.
    """
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        label_weights: Dict,
        language_weights: Dict,
        pos_weight: float,
        Config,
):
    """
    Trains the model for one epoch on the given training data.

    The function computes the loss for each batch, applies weighting factors to balance the
    labels and languages, and updates the model parameters using the gradient descent
    optimization process. It uses FocalLoss for addressing class imbalance and manages the
    learning rate adjustment via the provided scheduler.

    Args:
        model: The model to train.
        train_loader: DataLoader providing the training dataset.
        optimizer: Optimizer for the training process, e.g., Adam or SGD.
        scheduler: Learning rate scheduler to update the learning rate after each epoch.
        label_weights (Dict): A dictionary mapping labels to their respective weights for
            loss computation.
        language_weights (Dict): A dictionary mapping language IDs to their respective
            weights for loss computation.
        pos_weight (float): Weight applied to positive labels to manage class imbalance.
        Config: Configuration object containing settings such as `DEVICE` for specifying
            training on GPU or CPU.

    Returns:
        float: The average weighted loss value over the training epoch.
    """
    model.train()
    total_loss = 0
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = FocalLoss(gamma=2)

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        labels = batch["labels"].to(Config.DEVICE)
        languages = batch["language"]

        optimizer.zero_grad()

        # Get logits (not loss)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Compute per-sample loss
        loss_per_sample = criterion(logits, labels)

        # Compute weights
        label_weight_batch = torch.tensor(
            [label_weights[label.item()] for label in labels],
            dtype=torch.float,
            device=Config.DEVICE,
        )

        language_weight_batch = torch.tensor(
            [language_weights[lang] for lang in languages],
            dtype=torch.float,
            device=Config.DEVICE,
        )

        combined_weights = label_weight_batch * language_weight_batch
        combined_weights = combined_weights / combined_weights.mean()

        weighted_loss = (loss_per_sample * combined_weights).mean()

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += weighted_loss.item()
        progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

    scheduler.step()

    return total_loss / len(train_loader)


def validate(model, val_loader, Config):
    """
    Validates the performance of the model on a validation dataset. Calculates the average
    loss and collects predictions, true labels, and language metadata.

    Args:
        model: The neural network model to validate.
        val_loader: DataLoader providing the validation dataset in batches.
        Config: Configuration object containing model parameters and device information.

    Returns:
        Tuple containing:
            - avg_loss (float): The average loss across all validation batches.
            - all_preds (List[int]): A list of predicted class indices for the validation dataset.
            - all_labels (List[int]): A list of true class indices for the validation dataset.
            - all_languages (List[str]): A list of language metadata for each observation
              in the validation dataset.
    """
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []
    all_languages = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)
            languages = batch["language"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            # criterion = torch.nn.CrossEntropyLoss()
            criterion = FocalLoss(gamma=2)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_languages.extend(languages)

    avg_loss = total_loss / len(val_loader)

    return avg_loss, all_preds, all_labels, all_languages
