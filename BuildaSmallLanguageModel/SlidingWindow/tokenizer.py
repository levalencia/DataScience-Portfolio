# tokenizer.py
class SimpleTokenizer:
    def __init__(self, text):
        # Split text into words
        self.words = text.split()
        # Assign each unique word a token ID
        self.tokens = {word: idx for idx, word in enumerate(set(self.words))}
    
    def tokenize(self):
        """Convert words to token IDs"""
        return [self.tokens[word] for word in self.words]
    
    def decode(self, token_ids):
        """Convert token IDs back to words"""
        reverse_tokens = {idx: word for word, idx in self.tokens.items()}
        return [reverse_tokens[token] for token in token_ids]

# Example usage
if __name__ == "__main__":
    text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
    tokenizer = SimpleTokenizer(text)
    token_ids = tokenizer.tokenize()
    print("Token IDs:", token_ids)
