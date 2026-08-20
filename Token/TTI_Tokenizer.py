from tokenizers import Tokenizer, AddedToken


class TTITokenizer:
    def __init__(self, tokenizer_path="Token/bpe_tokenizer_20k.json", max_length=64):
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        if self.tokenizer.token_to_id("[PAD]") is None:
            self.tokenizer.add_special_tokens([AddedToken("[PAD]", special=True)])

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(length=self.max_length,pad_id=self.pad_id,pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=self.max_length)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text)

    def encode_ids(self, text):
        encoded = self.encode(text)
        return {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask
        }

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save(self, path):
        self.tokenizer.save(path)