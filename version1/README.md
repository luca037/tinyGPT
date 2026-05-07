# Recreating GPT-2 (124M)

This project is an improvement of `version0`. Here we try to recreate the GPT-2 (124M) architecture. This implementation is fully explained at 
[this link](https://www.youtube.com/watch?v=l8pRSuU81PU&t=2614s).

## MultiHead Self-Attention

In this implementation the multi-head self-attention is implemented by class `CausalSelfAttention`. In previous implementation we used two classes: `Head` and `MultiHeadAttention`. Here we've just merged the two. However the forward function 

**The names of the variables cannot be modified.** Those one are used so that we can easily import the pre-trained weights that we can find from Huggingface. So one can either choose to pre-train from scratch, or just load them.

#### Note1: unpack q, k, v

```python
qkv = self.c_attn(x)
```

Consider that the input is $X\in\mathbb R^{T\times C}$ where $T$ is the *context size* and $C$ is the size of the emebddings. We're considering one single batch. The above line of code perform the following operation:

$$
qkv = X \times W = [Q | K | V] \in \mathbb R^{T\times 3C}
$$
$$
Q,K,V\in\mathbb R^{T\times C}
$$
where $W\in\mathbb R^{C\times 3C}$ is the matrix of weights. Finally we access the query matrix $Q$, key matrix $K$ and value matrix $V$ by splitting $qkv$:

```python
q, k, v = qkv.split(self.n_embd, dim=2)
```

#### Note2: reshape and compute attention

```python
k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
```

Recall that we're implementing multi-head self attention. In `version0` we've seen that the output of each head is concatenated.

In this implementation the query, key, value matrix of each head is grouped inside a big matrix.  Before executing the lines of code above, `k` looks like this

```math
k = 
\begin{bmatrix}
K_1 | \cdots | K_{nh}
\end{bmatrix} \in \mathbb R^{T\times C},
\quad K_i \in \mathbb R^{T\times hs}
```

```math
\begin{split}
nh &:= \text{number of heads}\\
hs &:= \text{head size}
\end{split}
```

where $hs = C \mod nh$. After the execution of the code above, each matrix $K_i$ is placed in the batch dimension (the third dimension), so that `k` becomes a `(nh, T, C)` tensor (ignore batch dimension `B`).

**Why we do this mad trick?** Is computational more efficient. We need to perform the reshape in order to compute self-attention (for each head) using `F.scaled_dot_product_attention()` function. Finally we reshape to get the output of multi-head self-attention: a tensor of shape `(B, T, C)`.

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
y = y.transpose(1, 2).contiguous().view(B, T, C)
```


## GPT2 architecture

#### Note3: sharing weights

The unembedding matrix $V$ is the transpose of the embedding matrix $U$

```math
V = U^T
```
In our implementation this is obtained with

```python
self.transformer.wte.weight = self.lm_head.weight
```

We can check that the two matrices share the same weights with the following instructions:

```python
# Do they share the same values?
w1 = self.transformer.wte.weight 
w2 = self.lm_head.weight
print((w1 == w2).all())

# Do they have the same pointer?
ptr1 = self.transformer.wte.weight.data_ptr()
ptr2 = self.lm_head.weight.data_ptr()
print(ptr == ptr2)
```

Both expression should return `true`.

**But why doing so?** This strategy was introduced in [this](https://arxiv.org/pdf/1608.05859) paper and was used in the famous *Attention is all you need* paper. In the former, they argue that $U$ and $V$ play a similar role:

> *We call $U$ the input embedding, and $V$ the output embedding. In both matrices, we expect rows that correspond to similar words to be similar: for the input embedding, we would like the network to react similarly to synonyms, while in the output embedding, we would like the scores of words that are interchangeable to be similar.*

By enforcing $U =V$ we get better results and also we have less trainable parameters (a tons less)!
**Training both.** With this choice, during training, we're effectively updating twice the weights used in embedding and unembedding matrices. The first update is performed from the top and the second one is performed from the bottom (backpropagation of error).

## Make the code fast!

In this section we introduce some tricks one can use to speed up our code by fully utilizing our hardware.

#### Note4: FP32, TF32, BF16

The data type that we use to represent the parameters in our network play a role in computational cost. The trade-off we face is between **precision** and **memory usage**.

<div align="center">
Table: NVIDIA A100 specs.
<img width="700" height="600" alt="Pasted image 20260427180159" src="https://github.com/user-attachments/assets/8bc03d6f-8793-4d7f-87ac-02c9859fe28e" />
</div>

Pytorch by default uses FP32 and this is inefficient for our application! By casting our parameters to BF16 we increase the **TFLOPS** from 19 to 321!

**What do we lose by using BF16?** First note that not all GPU support all data types. For example, I have a NVIDIA GEFORCE RTX 4070 Super and it only supports FP64, FP32, FP16. That's because it was not designed to train deep learning models but for gaming.

<div align="center">
<img width="400" height="300" alt="Pasted image 20260427182009" src="https://github.com/user-attachments/assets/72d8907a-1b57-4d8e-9c44-7d02f7ad086e" />
</div>

BF16 have the same range as FP32 and TF32 but the mantissa occupies less bits, meaning that we have a loss in precision.  

We don't really care about the precision loss. To compensate this we train for longer, but essentially we're not losing anything in terms of performances of our final model. (Note that this is true for our application, not in general!).

**TF32.** In theory, using TF32 can provide up to an 8× speedup compared to FP32. Empirically we observe that we don't actually get an 8x speedup, this is because we're still shipping to our GPU FP32 tensors! Those tensors are then converted (via hardware) to TF32 and the matrix multiplication results faster.

In PyTorch we can choose to use TF32 with a single line of code:

```python
torch.set_float32_matmul_precision('high')
```

Note: as mentioned if you print the dtype of the tensor you should still see `torch.float32`. If code doesn't speed up, then it might be that your GPU doesn't support TF32.

**BF16.** It is not recommended to use BF16 for every tensor in your code because certain operations (like Softmax, LayerNorm, or calculating Loss) are numerically unstable in half-precision and will result in `NaN` values. In practice we use **mixed precision package** that allows us to automatically cast to BF16 when performing certain operations. From the [doc](https://docs.pytorch.org/docs/stable/amp.html#torch.autocast):

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with torch.autocast(
	    device_type=DEVICE, dtype=torch.bfloat16
	):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    optimizer.step()
```

That is, we use the autocast during forward pass! Again, if you don't see any improvement => your GPU doesn't support it.

#### Note5: torch.compile

`torch.compile` is used to speedup our code. From the [doc](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html):

> *Speedup mainly comes from reducing Python overhead and GPU read/writes.*

`torch.complie` analyze our entire code and tries to optimize some operations.

<div align="center">
Figure: GPU-CPU communication.
  <img width="700" height="450" alt="Pasted image 20260428104105" src="https://github.com/user-attachments/assets/5f6ac100-561c-4b5b-b869-d23817d8684f" />
</div>

<div align="center">
Figure: Zoom in GPU architecture. An SM is a Streaming Multiprocessor: those are used to perform the computation and they have an internal memory (cache)<img width="2000" height="897" alt="Pasted image 20260428103758" src="https://github.com/user-attachments/assets/12225a54-df35-4887-b51b-545f34f83e95" />  
</div>

The idea is simple: there are some operation in your code that requires multiple read and write operations performed in the **HBM**. Those operations can be optimized if we use the cache memory of our GPU. `torch.compile` is used to reduce the read/writes in HBM and to rely more on the cache.

#### Note6: flash attention

In `version0` we've implemented the attention operation using the definition, instead in this version we're using a single line of code:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

It was introduced in [this](https://arxiv.org/abs/2205.14135) paper. We will not go into the details here, but it is a rewrite of the attention algorithm. The idea is that the $N\times N$ attention matrix is never materialized in the HBM: to do so they perform **more FP operations** than the standard attention however they use **less read/writes ops** resulting in a faster algorithm.

#### Note6: use values with many power of 2!

Set parameters (like number of heads, size of embeddings, batch size, etc) to values that "have" as many as possible powers of two. This can make a huge difference in some cases. This is related to how CUDA works.

In our implementation the default `vocab_size` is 50257: we didn't choose this value, this comes from the encoder used for GPT-2. By replacing this value with 50304, we speed up execution.

**But wait, you're increasing the vocabulary size?** Yes. This is not a problem because:

- In input those encodings will never be used because the encoder generates values that are in the range $0,\cdots,50257$.
- In output the logits of each new token will be pushed towards $-\infty$ aka 0 probability!

Regarding the output, note that depending on the dataset used to train the model, many tokens may never appear; as a result, their corresponding logits are pushed toward $-\infty$ during training.

So we're safe, we're not breaking anything by padding our embedding matrix.

#### Results on my NVIDIA 4070 Super

Settings: batch size = 4, context window = 1024.

Starting with only flash attention enabled:

```
time for one training step: 235.58ms
token/sec: 17386
```

Enabling TF32:

```
time for one training step: 177.47ms
token/sec: 23080
```

Using mixed precision (BF16):

```
time for one training step: 102.44ms
token/sec: 39982
```

Enabling torch.compile:

```
time for one training step: 90.33ms
token/sec: 45345
```

Increasing vocab_size:

```
time for one training step: 89.25ms 
token/sec: 46003
```

## Configure the optimizer

#### Note7: normalize gradient

This is simply achieved with one line of code:

```pyhon
norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Its job is to prevent **exploding gradients** without modifying the gradient direction. We also store the `norm` of the gradient so that we can monitor it during training: this value should become stable (almost constant) as training procedes.

#### Note8: learning rate scheduler

In the original GPT-2 paper they've used the cosine decaying learning rate. One can decide to change the scheduler.

<div align="center">
<img width="639" height="510" alt="Pasted image 20260429175926" src="https://github.com/user-attachments/assets/c6166961-360c-4460-a14b-7b65d40e75ae" />
</div>

In the code this is implemented in a simple function that returns the learning rate based on the current step.

```python
def get_lr(step):
    """ Learning Rate scheduler """
    # (1) Linear warmup for WARMUP_ITERS steps.
    if step < WARMUP_STEPS:
        return MAX_LR * (step+1) / WARMUP_STEPS
    # (2) If step > MAX_STEPS, return min val.
    if step > MAX_STEPS:
        return MIN_LR
    # (3) In between, use cosine decay down 
    # to min learning rate.
    decay_ratio = (
	    (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
	)
    assert 0 <= decay_ratio <= 1
    # Coeff starts at 1 and goes to 0.
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return MIN_LR + coeff * (MAX_LR - MIN_LR)
```

#### Note9: configuring AdamW

The configurations are defined in `GPT.configure_optimizers`. In this function there are two interesting thing.

The first one is the **weight decay**. Recall that weight decay is a component added to the loss function:

```math
\mathcal L(\pmb w) + \frac \lambda 2\sum_{i}^{||\pmb w||}w_i^2
```

In the summation we don't want to consider biases and layernorm parameters, we just want to penalize the weights in the embedding matrix and in the `nn.Linear` layers. For this reason in the code we've divided the parameters in two groups

```python
# Define the group of decay and non-decay params.
decay_params = [
	p for n, p in param_dict.items() if p.dim() >= 2
]
nodecay_params = [ # Biases are 1-dimensional vectors.
	p for n, p in param_dict.items() if p.dim() < 2
]
optim_groups = [
	{'params': decay_params, 'weight_decay': weight_decay},
	{'params': nodecay_params, 'weight_decay': 0.0}
]
```

The second interesting thing is the `fused` parameter. Not all optimizer support this (check [doc](https://docs.pytorch.org/docs/2.11/optim.html#torch.optim.SGD)), but if you're using cuda, then you should enable it. From the doc:

> *A few of our optimizers have even faster fused implementations, which fuse the big chunks of computation into one kernel.*

High level view: when `fused=False` we're using either the `forloop` or `for-each`. Those two methods are slower because they require to run more kernels to perform all the updates of our parameters. With fused we *fuse* all those kernels into a single one. From the doc:

> *In general, the performance ordering of the 3 implementations is fused > foreach > for-loop*. 

**But what's a kernel in GPU context?** Check [this](https://modal.com/gpu-glossary/device-software/kernel) article.

**fused=False vs. fused=True.** On my my NVIDIA 4070 Super we get

```
time for one training step: 89.25ms 
token/sec: 46003
```

by setting `fused=False`. If we set it to `True`:

```
time for one training step: 78.69ms
token/sec: 52049
```

That's not bad!


#### Note10: gradient accumulation

In the original paper, the batch size of GPT-2 (124M) is 0.5 million tokens. In GPT-2 the context window size is $T = 1024$, so in order to have 0.5 million tokens for each batch, we should set $B = 0.5\cdot 10^6 / 1024 \approx 488$. This is infeasible: in my GPU $B=4, T=1024$ takes almost 6GB. $B=488$ would require a massive amount of memory.

To solve this issue we use **gradient accumulation**: it simply means that we compute many backward pass before updating the parameters of our model. In this way we sequentially load smaller batches in GPU memory; we perform the forward + backward pass for each of them; finally we compute the update using the accumulated gradients.

**How many backward pass we need?** This can be easily computed:

```python
total_batch_size = 2**19 # ~0.5M, in number of token.
B = 4    # Micro batch size.
T = 1024 # Sequence length.
assert total_batch_size % (B * T ) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}.")
print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}.")
```

**The normalization factor.** By default in PyTorch, when we compute the loss function over a batch, it returns the *mean loss* (`reduction='mean'`). 

For example if our loss is the MSE (for a binary classification task) and we compute the loss over a minibatch of 4 we get the following total loss:

```
total_loss = 1/4 * (
	(y[0] - yhat[0])**2 +
    (y[1] - yhat[1])**2 +
    (y[2] - yhat[2])**2 +
    (y[3] - yhat[3])**2
)
```

where `yhat` is the prediction of our model and `y` are the labels. In gradient accumulation we compute the loss over a **micro batch**. If we consider micro batch of size 1, we get the following loss:

```
L0 = (y[0] - yhat[0])**2
L1 = (y[1] - yhat[1])**2
L2 = (y[2] - yhat[2])**2
L3 = (y[3] - yhat[3])**2
total_loss = L0 + L1 + L2 + L3
```

`L0` to `L3` are computed sequentially. Note that the two `total_loss` are different! We're missing 1/4 factor. Let's add it:

```
L0 = 1/4 * (y[0] - yhat[0])**2
L1 = 1/4 * (y[1] - yhat[1])**2
L2 = 1/4 * (y[2] - yhat[2])**2
L3 = 1/4 * (y[3] - yhat[3])**2
total_loss = 1/4 (L0 + L1 + L2 + L3)
```

In our code the micro batch has size 4 and the batch size is 512 (we rounded 488 up to 512) => we perform 512/4 = 128 gradient accumulation steps. The total loss should be:

```
total_loss = 1/512 * (L0 + L1 + ... + L511)
```

This is our target. In each accumulation step the loss is:

```
Li = 1 / 4  * (y[0] - yhat[0])**2
```

Therefore we need to multiply each `Li` by 1/128.
