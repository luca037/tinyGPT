# tinyGPT

The project is an extremely simple implementation of GPT. All the details of the code are explained in [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&index=55&t=419s) by Andrej Karpathy.

In this project we implement a **transformer decoder** architecture for language modeling. The structure reported below.

<div align="center">
  <img width="681" height="600" alt="image" src="https://github.com/user-attachments/assets/4e031795-14b3-4e0d-b64d-ee5150c9fa1b" />
</div>

### Dataset and Tokens

The dataset used is called `tinyshakespeare`,  is a single `txt` file with 40K lines of Shakespeare.

You can use any dataset you want. In must be placed in `dataset.txt` file. In the example we're using the so called `tinyshakespeare` dataset taken from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

In this project the tokens are just single characters (=> our vocabulary is pretty small). So our model will learn how to *predict next character* given the ones in the context window.

### A Single Head of Self-Attention

**Goal:** compute self-attention equations:

```math
\text{Attention(Q, K, V)} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```
where:
```math
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
```
where $X$ is the input matrix: each row contains the embedding associated to each token in the context window.
```math
X = 
\begin{bmatrix}
emb(\text{token 0)})\\
\vdots\\
emb(\text{token $n$})
\end{bmatrix}
```
In the code this is performed in three different steps.

### MultipleHead Attention

**Goal:** compute `n_heads` self-attention and concatenate them.

```math
out = \left[ \cdots | \text{Attention}(Q_i, K_i, )\right|\cdots]
```
Let $k$ be the size of the embeddings, then usually we define the size of head head as $k/n$ where $n$ is the number of heads. So with this choice, the vector $out$ has size $k$.
