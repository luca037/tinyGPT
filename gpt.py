import torch
import torch.nn as nn
from torch.nn import functional as F

### GLOBAL VARS ###
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"INFO - Device = '{DEVICE}'")
INPUTFN = './dataset.txt' # Our dataset.

CONTEXT_SIZE = 256 # Size of the context window.
EMBD_SIZE = 384    # Embeddings vector size.
NUM_HEADS = 6
NUM_BLOCKS = 6

DROPOUT = 0.2
BATCH_SIZE = 64
EVAL_ITERS = 200    # Iterations to estimate train/val losses.
EVAL_INTERVAL = 500 # Evaluation is performed every once in a while.
MAX_ITER = 5000     # Total number of training steps.
LR = 3e-4           # Learnig Rate

torch.manual_seed(1337)

############################# ARCHITECTURE ####################################

class Head(nn.Module):
    """ A single head of self-attention """

    def __init__(self, head_size): 
        super().__init__()

        self.query = nn.Linear(EMBD_SIZE, head_size) # q = Wq * x
        self.key   = nn.Linear(EMBD_SIZE, head_size) # k = Wk * x
        self.value = nn.Linear(EMBD_SIZE, head_size) # v = Wv * x

        # This is used to apply the mask.
        # register_buffer is  used to register a buffer 
        # that should not be considered a model parameter.
        self.register_buffer( 
            'tril',
            torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Input shape:  (batch, time-step, embd_size) aka (B, T, E)
        # Output shape: (batch, time-step, head_size) aka (B, T, hs)

        # Input shape.
        B, T, E = x.shape

        # Compute query, key, value
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)    # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Compute attention scores.
        # 1. Compute dot products: q x k^T
        wei = q @ k.transpose(-2, -1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # 2. Apply mask to preserve causality.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # 3. Normalize with softmax along last dim.
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # 3.2 Apply dropout.
        wei = self.dropout(wei) # (B, T, T)

        # Finally the head's output.
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, n_heads, head_size):
        super().__init__()

        # Define a list of heads.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

        # Define a projection.
        self.proj = nn.Linear(head_size * n_heads, EMBD_SIZE)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # k := head_size * n_heads
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, k)
        out = self.proj(out) # (B, T, E)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """ Simple FFNN """

    def __init__(self, n_embd):
        super().__init__()
        
        # The dimensions are taken from original 
        # `Attention is all you need` paper.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ A transformer block """

    def __init__(self, n_embd, n_heads):
        super().__init__()

        # Define size of each head.
        head_size = n_embd // n_heads

        self.ln1 = nn.LayerNorm(n_embd)
        self.maa = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Here we need to consider the residual stream.
        x = x + self.maa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """ GPT language model: complete architecture """

    def __init__(self, n_embd, vocab_size, context_size, n_blocks, n_heads):
        super().__init__()

        # Mapping: token index -> embedding.
        self.token_embd_table = nn.Embedding(vocab_size, n_embd)
        # Mapping: token position -> embedding.
        self.position_embd_table =  nn.Embedding(context_size, n_embd)

        # Transformer blocks.
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads) for _ in range(n_blocks)
        ])

        # Final layer norm (usually added).
        self.ln_f = nn.LayerNorm(n_embd)

        # Unembedding (language model head).
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx and targets are tensor of integers -> size (B, T).
        B, T = idx.shape

        # 1. Compute token embd + pos embd
        token_embd = self.token_embd_table(idx) # (B, T, E)
        pos_embd = self.position_embd_table(torch.arange(T, device=DEVICE)) # (T, E)
        x = token_embd + pos_embd # (B, T, E) (pytorch manage the mismatch).

        # 2. Pass though many blocks.
        x = self.blocks(x) # (B, T, E)

        # 3. Layer norm + logits.
        x = self.ln_f(x) # (B, T, E)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # 4. Compute loss, if necessary.
        loss = None
        if targets is not None: 
            # Reshape needed because of the loss func (see doc).
            B, T, E = logits.shape
            logits = logits.view(B * T, E)
            targets = targets.view(B * T)
            # Compute loss.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        # idx is a (B, T) array
        for _ in range(max_tokens):
            # Crop to last contet window.
            idx_cond = idx[:, -CONTEXT_SIZE:]
            # Get logits.
            logits, _ = self(idx_cond) # (B, T, vocab_size)
            # Get last time-step.
            logits = logits[:, -1, :] # (B, vocab_size)
            # Apply softmax.
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Sample from distribution.
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sample.
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


############################## UTILS ##########################################

def get_tokenizer(text):
    """ Return tokenizer and detokenizer """
    # All unique chars that occours in the input.
    chars = sorted(list(set(text)))

    stoi = { c:i for i, c in enumerate(chars) }
    itos = { i:c for i, c in enumerate(chars) }

    # Tokenizer: maps string -> list of int
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return chars, encode, decode


def get_batch(data):
    """ Returns a batch sampled from data """

    # Define the first index for each batch.
    idx = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))

    # Stack input samples.
    x = torch.stack([data[i:i+CONTEXT_SIZE] for i in idx])

    # Labels are just input shifted by 1.
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in idx])

    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


@torch.no_grad()
def estimate_loss(model, train, val):
    out = {} # Store both train and validation losses.
    model.eval()
    for key, split in zip(['train', 'val'], [train, val]):
        losses = torch.zeros(EVAL_ITERS) # Store evaluations.
        # Estimate loss (mean).
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[key] = losses.mean()
    model.train()
    return out


def train_step(model, optim, train):
    xb, yb = get_batch(train)
    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()


############################### MAIN ##########################################

def main():
    # Read input data.
    with open(INPUTFN, 'r') as f: text = f.read()

    # Encode input data.
    vocab, encode, decode = get_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)

    # Define train/val split.
    n = int(0.9 * len(data))
    train = data[:n]
    val = data[n:]

    # Define model and optimizer.
    model = GPTLanguageModel(
        n_embd=EMBD_SIZE, 
        vocab_size=len(vocab), 
        context_size=CONTEXT_SIZE, 
        n_blocks=NUM_BLOCKS, 
        n_heads=NUM_HEADS
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for iter in range(MAX_ITER):
        # Estimate loss every once in a while.
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITER - 1:
            losses = estimate_loss(model, train, val)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Train model.
        train_step(model, optimizer, train)

    # Generate tokens.
    # Init context with characther with idx 0 -> '\n'.
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    text = decode(model.generate(context, 5000)[0].tolist())

    print("Generated text:\n", text)
    with open("output.txt", 'w') as f: f.write(text)


if __name__ == '__main__':
    main()
