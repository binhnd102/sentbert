import pickle
import re
import joblib

class Tokenizer:
    def __init__(self, stoi_file):
        self.stoi = joblib.load(stoi_file)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.num_vocab = len(self.stoi)
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.oov = "<unk>"
    
    def encode(self, text, max_length=128, padding="max_length"):
        tokens = []
        for part in re.split(f"({'|'.join([self.mask_token, self.oov, self.pad_token])})", text):
            if part in [self.mask_token, self.oov, self.pad_token]:
                tokens.append(self.stoi.get(part, self.stoi[self.pad_token]))
            else:
                tokens.extend([self.stoi.get(s, self.stoi[self.oov]) for s in part])
        if padding == "max_length":
            tokens = tokens[:max_length] + [self.stoi[self.pad_token]] * (max_length - len(tokens))
            return tokens
        return tokens
    
    def decode(self, tokens):
        return " ".join([self.itos[t] for t in tokens])
    
    @property
    def speical_tokens(self):
        return [self.pad_token, self.mask_token, self.oov]
    
    @property
    def pad_token_id(self):
        return self.stoi[self.pad_token]
    
    def get_non_special_vocab(self):
        specials = self.speical_tokens
        return [self.itos[i] for i in range(self.num_vocab) if self.itos[i] not in specials]

    def get_vocab(self):
        return [self.itos[i] for i in range(self.num_vocab)]

tokenizer = Tokenizer("data/chotot/stoi.pkl")