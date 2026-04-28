from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import transformers
import tiktoken

import time

############################# GOBAL VARS ######################################

DEVICE = 'cuda' 
if torch. cuda. is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
print(f"using device: {DEVICE}")

############################# ARCHITECTURE ####################################

class CausalSelfAttention(nn.Module):
    """ Implementation of MultiHead self-attention """

    def __init__(self, config):
        super().__init__()

        # Needed because we concatenate the output of each head.
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads, but in a batch.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection: applied after multi-head sefl-att.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 

        # Dimension of the embeddings and number of heads.
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # (Batch size, sequence length, embedding dimensionality (n_embd)).
        B, T, C = x.size() 

        # Calculate query, key, values for all heads in batch and 
        # move head forward to be the batch dim.
        # Names: 
        #  nh is "number of heads", 
        #  hs is "head size",
        #  C (number of channels) = nh * hs
        qkv = self.c_attn(x) # (B, T, 3 * n_embd) [see Note1 in README.md]

        # Define query, key, value matrix. Each matrix has size (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2) # dim=2 aka (3 * n_embd)

        # Magic trick [see Note2 in README.md].
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Compute self attention for each head.
        # Flash attention implementation. [see Note6 in README.md]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head 
                                                         # outputs side by side 
                                                         # (B, T, C)
        # Output projection.
        y = self.c_proj(y) # (B, T, C)
        return y

class MLP(nn.Module):
    """ FFNN layer """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # Because original GPT-2 uses it.
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """ A single transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """ Used to set architecture configurations """
    block_size: int = 1024  # Max sequence length.
    vocab_size: int = 50257 # Number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token.
    n_layer: int = 12       # Number of layers.
    n_head: int = 12        # Number of heads.
    n_embd: int = 768       # Embedding dimension.


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embeddings.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position embeddings.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Transformer blocks.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final linear transformation.
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language modeling head.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme. [see Note3 in README.md]
        self.transformer.wte.weight = self.lm_head.weight

        # Init params.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Default init used by GPT2 paper """
        if isinstance(module, nn.Linear):
            std = 0.02
            # TODO: i don't get this. See video minute 1h16min
            if hasattr(module, 'NANOGPT_SCALE_INIT'): 
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # Idx is of shape (B, T).
        B, T = idx.size()
    
        # Input check.
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Sum token embd and position embd.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Shape (T).
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (T, n_embd).
        tok_emb = self.transformer.wte(idx) # Token embeddings of shape (B, T, n_embd).
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer.
        for block in self.transformer.h:
            x = block(x) # (B, T, C)

        # Forward the final layernorm.
        x = self.transformer.ln_f(x) # (B, T, C)
        
        # Get logits from language modeling head.
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Compute loss, if necessary.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1) # (B*T,)
        )
        return logits, loss

    def generate(self, input, max_tokens, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        import tiktoken

        # Encode input.
        enc = tiktoken.get_encoding(model_type)
        tokens = enc.encode(input)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0) # (B, T, C)
        x = tokens.to(DEVICE)

        # Generate.
        while x.size(1) < max_tokens:
            with torch.no_grad():
                logits, _ = self(x) # (B, T, vocab_size)
                # Take the logits at the last position.
                logits = logits[:, -1, :] # (B, vocab_size)
                # Get the probabilities.
                probs = F.softmax(logits, dim=-1)
                # Do top-k sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # Select a token from the top-k probabilities.
                ix = torch.multinomial(topk_probs, 1) # (B, 1)
                # Gather the corresponding indices.
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # Append to the sequence.
                x = torch. cat ((x, xcol), dim=1)
        # print the generated text
        out = []
        for i in range(x.size(0)):
            tokens = x[i, :30].tolist()
            decoded = enc.decode(tokens)
            out.append(decoded)
        return out


    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from huggingface (IGNORE THIS) """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    #def configure_optimizers(self, weight_decay, learning_rate, device_type):
    #    # start with all of the candidate parameters (that require grad)
    #    param_dict = {pn: p for pn, p in self.named_parameters()}
    #    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    #    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #    optim_groups = [
    #        {'params': decay_params, 'weight_decay': weight_decay},
    #        {'params': nodecay_params, 'weight_decay': 0.0}
    #    ]
    #    num_decay_params = sum(p.numel() for p in decay_params)
    #    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #    if master_process:
    #        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #    # Create AdamW optimizer and use the fused version if it is available
    #    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #    use_fused = fused_available and device_type == "cuda"
    #    if master_process:
    #        print(f"using fused AdamW: {use_fused}")
    #    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    #    return optimizer

############################ DATALOADERS ######################################

class SimpleDataLoader:

    def __init__(self, B, T):
        # Batch size and context length.
        self.B = B
        self.T = T

        # Load the tiny shakespeare dataset.
        text = ""
        with open("./data/input.txt", 'r') as f: text = f.read()

        # Encode it.
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        # Print some info.
        print(f"Loaded {len(self.tokens)} tokens.")
        print(f"Each batch has {B * T} tokens.")
        print(f"1 epoch is {len(self.tokens) // (B * T)} batches.")

        self.idx = 0 # Posiiton index.
        
    def next_batch(self):
        buf = self.tokens[self.idx : self.idx + self.B * self.T + 1]
        x = (buf[:-1]).view(self.B, self.T) # Inputs.
        y = (buf[1:]).view(self.B, self.T) # Targets.
        # Update position index.
        self.idx += self.B * self.T
        # Reset index, if necessary.
        if self.idx + (self.B * self.T + 1) > len(self.tokens): self.idx = 0
        return x, y

############################## UTILS #########################################

def train_test():
    """ Make sure we can overfit a single batch """

    # Get a single batch.
    dl = SimpleDataLoader(5, 4)
    x, y = dl.next_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)

    # Crate default model.
    model = GPT(GPTConfig()).to(DEVICE)

    # Check loss with initialized weights.
    import math
    print("Loss should be around:", -math.log(1/model.config.vocab_size)) # Uniform distrib.
    _, loss = model(x, y)
    print("Loss is:", loss.item())

    # Lets overfit our batch to see if everything is working.
    optim = torch.optim.AdamW(model.parameters())
    for i in range(20):
        _, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"itr: {i}, loss: {loss.item()}")


def optim_tests():
    """ Tests just to see the benefit of each trick """
    B, T = 4, 1024 # If it doesn't fit to GPU memory => decrease B.

    # No optimization (only flash attention).
    def train_v1():
        model = GPT(GPTConfig()).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters())
        dl = SimpleDataLoader(B, T)
        for i in range(10):
            t0 = time.time() 
            x, y = dl.next_batch()
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, loss = model(x, y)
            print(logits.dtype)
            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000 # Time in milliseconds.
            tokens_per_sec = (dl.B * dl.T) / (t1 - t0)
            print(f"itr: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, token/sec: {tokens_per_sec}")
    #train_v1()

    # Set data type to TF32. [see Note4 in README.md]
    torch.set_float32_matmul_precision('high') # This will work depending on your GPU!
    #train_v1()

    # Version with auto-cast. [see Note4 in README.md]
    def train_v2():
        model = GPT(GPTConfig()).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters())
        dl = SimpleDataLoader(B, T)
        for i in range(10):
            t0 = time.time() 
            x, y = dl.next_batch()
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Auto-cast!
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                print(logits.dtype)
            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000 # Time in milliseconds.
            tokens_per_sec = (dl.B * dl.T) / (t1 - t0)
            print(f"itr: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, token/sec: {tokens_per_sec}")
    #train_v2()

    # Added torch.compile. [see Note5 in README.md]
    def train_v3():
        model = GPT(GPTConfig()).to(DEVICE)
        # Complie model.
        model = torch.compile(model)
        optim = torch.optim.AdamW(model.parameters())
        dl = SimpleDataLoader(B, T)
        for i in range(10):
            t0 = time.time() 
            x, y = dl.next_batch()
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Auto-cast!
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000 # Time in milliseconds.
            tokens_per_sec = (dl.B * dl.T) / (t1 - t0)
            print(f"itr: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, token/sec: {tokens_per_sec}")
    #train_v3()

    # Remove the ugly number (vocab_size). [see Note6 in README.md]
    def train_v4():
        model = GPT(GPTConfig(vocab_size=50304)).to(DEVICE)
        # Complie model.
        model = torch.compile(model)
        optim = torch.optim.AdamW(model.parameters())
        dl = SimpleDataLoader(B, T)
        for i in range(10):
            t0 = time.time() 
            x, y = dl.next_batch()
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Auto-cast!
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000 # Time in milliseconds.
            tokens_per_sec = (dl.B * dl.T) / (t1 - t0)
            print(f"itr: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, token/sec: {tokens_per_sec}")
    train_v4()


############################## MAIN ###########################################

def main():
    # Run some tests.
    #train_test()
    optim_tests()
    return

    #model = GPT.from_pretrained("gpt2")
    model = GPT(GPTConfig())
    model.eval()
    model.to(DEVICE)

    print("\n\nINFO - Generating output...")
    out = model.generate(input="Hi there!", max_tokens=100, model_type="gpt2")
    print("Generated output:\n")
    print(out[0])


if __name__ == "__main__":
    main()
