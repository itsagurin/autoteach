import os
import sys
import argparse
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tokenizer import MathTokenizer
from src.models.nanogpt import MicroNanoGPT
from src.engine.masking import prepare_batch
from src.tasks import get_task_for_level
from src.telemetry.metrics import compute_exact_match, compute_weight_norm


def evaluate_checkpoint(checkpoint_path: str, level: int):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    with open("configs/student_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer = MathTokenizer()
    model = MicroNanoGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_layer=config["model"]["n_layer"],
        n_head=config["model"]["n_head"],
        block_size=config["model"]["block_size"]
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    task = get_task_for_level(level)
    train_data, val_data = task.get_data()

    x_val, y_val = prepare_batch(val_data, tokenizer, device)
    x_train, y_train = prepare_batch(train_data, tokenizer, device)

    with torch.no_grad():
        val_logits, val_loss = model(x_val, targets=y_val)
        val_em = compute_exact_match(val_logits, y_val)

        train_logits, train_loss = model(x_train, targets=y_train)
        train_em = compute_exact_match(train_logits, y_train)

        w_norm = compute_weight_norm(model)

    print("\n" + "=" * 50)
    print(f"Evaluation Results for Level {level} ({task.name}):")
    print(f"  Train Loss:       {train_loss.item():.4f}")
    print(f"  Train EM:         {train_em:.2%}")
    print(f"  Val Loss:         {val_loss.item():.4f}")
    print(f"  Val EM:           {val_em:.2%}")
    print(f"  Weight Norm ||W||: {w_norm:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on task level.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_level_1.pt")
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        evaluate_checkpoint(args.checkpoint, args.level)
    else:
        print(f"Checkpoint {args.checkpoint} not found.")
