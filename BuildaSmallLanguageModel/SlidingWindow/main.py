# main.py
from tokenizer import SimpleTokenizer
from dataset import SlidingWindowDataset
from torch.utils.data import DataLoader

# Sample text
text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry."

# Step 1: Tokenize the text
tokenizer = SimpleTokenizer(text)
tokenized_text = tokenizer.tokenize()
print("Tokenized text:", tokenized_text)

# Step 2: Create the dataset
window_size = 4
stride = 1
dataset = SlidingWindowDataset(tokenized_text, window_size=window_size, stride=stride)

# Step 3: Load the data using PyTorch's DataLoader (batch_size=1 for simplicity)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Step 4: Iterate through the dataloader and print the input-output pairs
print("\nSliding Window Input-Output Pairs:")
for i, (input_window, next_token) in enumerate(dataloader):
    decoded_input = tokenizer.decode(input_window.squeeze().tolist())
    decoded_token = tokenizer.decode([next_token.item()])
    print(f"Sample {i+1}: Input: {decoded_input} -> Predict: {decoded_token}")
