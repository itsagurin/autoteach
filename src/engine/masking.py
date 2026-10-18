import torch
from src.models.tokenizer import MathTokenizer

def prepare_batch(
        equations: list[tuple[str, str]],
        tokenizer: MathTokenizer,
        device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input_ids = []
    batch_target_ids = []

    for prompt, answer in equations:
        prompt_tokens = tokenizer.encode(prompt)
        answer_tokens = tokenizer.encode(answer) + [tokenizer.eos_id]

        full_seq = prompt_tokens + answer_tokens

        x = full_seq[:-1]
        y = full_seq[1:]

        prompt_mask_boundary = len(prompt_tokens) - 1
        masked_y = [-100 if i < prompt_mask_boundary else token for i, token in enumerate(y)]

        batch_input_ids.append(x)
        batch_target_ids.append(masked_y)

    max_len = max(len(seq) for seq in batch_input_ids)

    padded_inputs = [seq + [tokenizer.pad_id] * (max_len - len(seq)) for seq in batch_input_ids]
    padded_targets = [seq + [-100] * (max_len - len(seq)) for seq in batch_target_ids]

    return (
        torch.tensor(padded_inputs, dtype=torch.long, device=device),
        torch.tensor(padded_targets, dtype=torch.long, device=device)
    )