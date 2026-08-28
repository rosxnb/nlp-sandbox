"""
Custom RNN cell built from scratch (no nn.RNN) to make every parameter and every
gradient explicit. Processes a sequence of T = 6 time steps, computes a scalar
loss on the final hidden state, calls .backward(), and prints the parameter
gradients produced by BPTT.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

INPUT_SIZE = 4
HIDDEN_SIZE = 3
SEQ_LEN = 6  # T >= 5 as required


class CustomRNNCell(nn.Module):
    """
    Implements the recurrence h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    using raw nn.Parameter tensors so the shared-weight structure is visible.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Same three parameter tensors are reused at every time step.
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        return torch.tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h)


def run_sequence(cell, inputs, h0):
    """Unrolls the cell across the sequence, storing every hidden state."""
    h = h0
    hidden_states = [h]
    for t in range(inputs.shape[0]):
        h = cell(inputs[t], h)
        hidden_states.append(h)
    return hidden_states  # [h0, h1, h2, ..., hT]


def main():
    cell = CustomRNNCell(INPUT_SIZE, HIDDEN_SIZE)

    # A random sequence of T input vectors.
    inputs = torch.randn(SEQ_LEN, INPUT_SIZE)
    h0 = torch.zeros(HIDDEN_SIZE)

    # ---- Forward pass through the unrolled graph ----
    hidden_states = run_sequence(cell, inputs, h0)
    h_final = hidden_states[-1]

    # ---- Scalar loss on the terminal hidden state ----
    # e.g. sum-of-squares "energy" of h_T, a stand-in for a real task loss.
    loss = (h_final ** 2).sum()

    print(f"Sequence length T = {SEQ_LEN}")
    print(f"Final hidden state h_{SEQ_LEN}: {h_final.detach().numpy()}")
    print(f"Scalar loss L = {loss.item():.6f}\n")

    # ---- Backpropagation Through Time ----
    loss.backward()

    print("Parameter gradients after .backward() (BPTT-accumulated):\n")
    for name, param in cell.named_parameters():
        print(f"{name}  shape={tuple(param.shape)}")
        print(param.grad, "\n")


if __name__ == "__main__":
    main()
