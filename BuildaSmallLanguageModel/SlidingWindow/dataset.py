# dataset.py
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, tokenized_text, window_size, stride=1):
        self.tokenized_text = tokenized_text
        self.window_size = window_size
        self.stride = stride
        self.input_windows, self.output_tokens = self._generate_windows()
    
    def _generate_windows(self):
        """Generate input-output pairs using a sliding window approach."""
        input_windows = []
        output_tokens = []
        for i in range(0, len(self.tokenized_text) - self.window_size, self.stride):
            # Input window of size `window_size`
            input_windows.append(self.tokenized_text[i:i + self.window_size])
            # The next token (the one we want to predict)
            output_tokens.append(self.tokenized_text[i + self.window_size])
        return input_windows, output_tokens
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.input_windows)
    
    def __getitem__(self, idx):
        """Return a single input-output pair."""
        return torch.tensor(self.input_windows[idx]), torch.tensor(self.output_tokens[idx])

# Example usage
if __name__ == "__main__":
    tokenized_text = [2, 4, 5, 6, 7, 1, 3, 8, 9, 10, 11, 12]  # Example tokenized text
    dataset = SlidingWindowDataset(tokenized_text, window_size=4, stride=2)
    for i in range(len(dataset)):
        input_window, next_token = dataset[i]
        print(f"Input: {input_window}, Next token: {next_token}")
