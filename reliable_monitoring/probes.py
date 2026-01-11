import numpy as np
import torch
import tqdm
from models_under_pressure.interfaces.dataset import LabelledDataset
from sklearn.linear_model import LogisticRegression

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def batched_average_over_sequence(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int = 256,
    device=device,
) -> torch.Tensor:
    """Compute average over sequence dimension in batches to save memory."""
    n_samples = activations.shape[0]
    averaged_acts = []
    for start_idx in tqdm.tqdm(range(0, n_samples, batch_size)):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_acts = activations[start_idx:end_idx].to(device)  # (batch, seq_len, hidden_dim)
        batch_mask = attention_mask[start_idx:end_idx].to(device)
        masked_acts = batch_acts * batch_mask.unsqueeze(-1)  # (batch, seq_len, hidden_dim)

        sum_acts = masked_acts.sum(dim=1)  # (batch, hidden_dim)
        lengths = batch_mask.sum(dim=1, keepdim=True)  # (batch, 1)
        avg_acts = sum_acts / lengths  # (batch, hidden_dim)
        averaged_acts.append(avg_acts.cpu())  # Move to CPU to save GPU memory

    return torch.cat(averaged_acts, dim=0)  # (n_samples, hidden_dim)


def train_probe(
    dataset: LabelledDataset,
) -> LogisticRegression:
    activations = dataset.other_fields["activations"]  # (n_samples, seq_len, hidden_dim)
    attention_mask = dataset.other_fields["attention_mask"]  # (n_samples, seq_len)
    labels = dataset.labels_numpy()

    # Compute average activations over sequence dimension
    avg_activations = batched_average_over_sequence(activations, attention_mask)  # (n_samples, hidden_dim)

    # Train logistic regression probe
    probe = LogisticRegression(max_iter=1000)
    probe.fit(avg_activations, labels)

    return probe


def probe_function(clf: LogisticRegression, dataset: LabelledDataset) -> np.ndarray:
    activations = dataset.other_fields["activations"]
    attention_mask = dataset.other_fields["attention_mask"]
    X = batched_average_over_sequence(
        activations,
        attention_mask,
        batch_size=512,
    ).numpy()
    probs = clf.predict_proba(X)[:, 1]
    return probs
