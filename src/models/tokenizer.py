class MathTokenizer:
    def __init__(self):
        chars = sorted(list("0123456789+-*=% "))
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

        self.vocab = [self.pad_token, self.eos_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.pad_id = self.stoi[self.pad_token]
        self.eos_id = self.stoi[self.eos_token]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos[t] for t in tokens if t not in (self.pad_id, self.eos_id)])