# Assignment 04 — Recurrent Neural Networks and Backpropagation Through Time

Source: [`RNN_Assignment_BPTT.pdf`](./RNN_Assignment_BPTT.pdf)

A written assignment on RNN mechanics and BPTT, submitted as handwritten notes, with one
implementation question answered in code.

## Contents

| File | Description |
|---|---|
| [`RNN_Assignment_BPTT.pdf`](./RNN_Assignment_BPTT.pdf) | The question paper |
| [`NLP - RNN Handwritten.pdf`](./NLP%20-%20RNN%20Handwritten.pdf) | **Deliverable** — scanned handwritten answers |
| [`rnn_bptt.py`](./rnn_bptt.py) | **Deliverable** — Question 8 implementation |

## Question 8 — Implementation

A custom RNN cell built from raw `nn.Parameter` tensors rather than `nn.RNN`, so that every parameter
and every gradient stays explicit. It unrolls the recurrence

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

over `T = 6` time steps, computes a scalar loss on the final hidden state, calls `.backward()`, and
prints the BPTT-accumulated parameter gradients.

### Running

```bash
../.venv/bin/python rnn_bptt.py
```

Requires only PyTorch, which the shared root environment already provides — see the
[root README](../README.md) for setup.

### Output

```
Sequence length T = 6
Final hidden state h_6: [0.40167424 0.00826048 0.1117903 ]
Scalar loss L = 0.173907

Parameter gradients after .backward() (BPTT-accumulated):

W_xh  shape=(3, 4)
tensor([[ 0.3234, -0.5755, -1.0442, -0.0196],
        [-0.0393,  0.0305, -0.0341,  0.0299],
        [ 0.1517, -0.2263, -0.3315, -0.0303]])

W_hh  shape=(3, 3)
tensor([[-0.0698,  0.0263,  0.0706],
        [-0.0050, -0.0063, -0.0069],
        [-0.0201,  0.0146,  0.0311]])

b_h  shape=(3,)
tensor([0.6085, 0.0556, 0.1639])
```

The run is seeded (`torch.manual_seed(0)`), so these values reproduce exactly.

### What the gradients represent

Each entry of `W_hh.grad`, `W_xh.grad` and `b_h.grad` is ∂L/∂(that parameter) — the total,
BPTT-accumulated sensitivity of the scalar loss to a one-unit change in that weight, **summed across
every time step where the weight was used** (per Q6). Because the same three tensors are reused at
every step, a single gradient entry aggregates contributions from all six unrollings. This is exactly
what an optimiser consumes: it nudges each parameter along `-grad` to reduce the loss.

**Connection to Q7 (vanishing gradients).** `W_hh.grad` has visibly smaller-magnitude entries than
`W_xh.grad`. That is the vanishing-gradient effect in miniature: gradient signal reaching `W_hh` is
repeatedly multiplied through the recurrence — once per time step — so contributions from early steps
are attenuated by the product of the Jacobians. With `tanh` saturating and weights initialised small,
that product shrinks geometrically in `T`. `W_xh` at step `t` sits closer to the loss and takes fewer
such multiplications, so its gradient survives better.
