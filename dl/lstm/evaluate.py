import string
import torch
from helper import *

all_characters = string.printable

def evaluate(model, prime_str='A', predict_len=100, temperature=0.8):
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state: todo is this working?
    for p in range(len(prime_str) - 1):
        model(prime_input[p])
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output = model(inp)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted