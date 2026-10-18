import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.tokenizer import MathTokenizer
from src.models.nanogpt import MicroNanoGPT
from src.engine.masking import prepare_batch


def test_tokenizer():
    tokenizer = MathTokenizer()
    text = "97*42=4074"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text, f"Decoded '{decoded}' != original '{text}'"
    assert tokenizer.pad_id != tokenizer.eos_id
    print("[OK] Tokenizer test passed")


def test_nanogpt_shapes_and_loss():
    tokenizer = MathTokenizer()
    model = MicroNanoGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layer=2,
        n_head=2,
        block_size=32
    )

    batch = [("12+34=", "46"), ("5*6=", "30")]
    device = torch.device("cpu")
    x, y = prepare_batch(batch, tokenizer, device)

    assert x.shape == y.shape
    logits, loss = model(x, targets=y)

    assert logits.shape == (x.shape[0], x.shape[1], tokenizer.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0
    print(f"[OK] Model forward & loss test passed: loss={loss.item():.4f}")


def test_engine_target_masking():
    tokenizer = MathTokenizer()
    batch = [("7+8=", "15")]
    device = torch.device("cpu")
    x, y = prepare_batch(batch, tokenizer, device)

    # Prompt tokens should be masked (-100) in target
    prompt_len = len(tokenizer.encode("7+8="))
    for i in range(prompt_len - 1):
        assert y[0, i].item() == -100, f"Prompt token at {i} was not masked: {y[0, i]}"

    # Answer tokens should not be masked
    assert y[0, prompt_len - 1].item() != -100
    print("[OK] Target masking test passed")


if __name__ == "__main__":
    test_tokenizer()
    test_nanogpt_shapes_and_loss()
    test_engine_target_masking()
    print("All model and engine tests passed successfully!")
