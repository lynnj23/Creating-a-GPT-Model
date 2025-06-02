#!/usr/bin/env python
# coding: utf-8

# # Exercise: Train your own transformer!
# 
# In this exercise, we will construct and train a minified GPT implementation. GPT refers to the "Generative Pre-trained Transformers" from OpenAI, originally described in ["Improving language understanding with unsupervised learning"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). This specific GPT implementation is heavily inspired by the [minGPT implementation](https://github.com/karpathy/minGPT) provided by [Andrej Karpathy](https://github.com/karpathy/).
# 
# There are three important PyTorch modules here:
# * `MultiHeadSelfAttention`: a self-attention implementation which will be provided for you;
# * `Block`: a transformer block which is repeated n_layer times in a GPT model;
# * `GPT`: the full GPT model itself, including intial embeddings, the GPT blocks, and the token decoding logic.
# 
# 
# The `GPT` module uses the `Block` module, which in turn uses the `MultiHeadSelfAttention` module.
# ```                                   
#     ┌────────────────────────┐     
#     │          GPT           │     
#     └────────────────────────┘     
#                 ▲                  
#     ┌───────────┴────────────┐     
#     │         Block          │     
#     └────────────────────────┘     
#                 ▲                  
#     ┌───────────┴────────────┐     
#     │ MultiHeadSelfAttention │     
#     └────────────────────────┘     
# 
# ```

# ## Step: Import and show MultiHeadSelfAttention

# In[1]:


from common import GPTConfig, MultiHeadSelfAttention

# Let's use a placeholder config to show how the attention layer works
config = GPTConfig(
    vocab_size=10,
    n_layer=3,
    n_embd=12,
    n_head=4,
    block_size=5,
)

attention = MultiHeadSelfAttention(config)

print(attention)


# ## Step: Create the Transformer Block
# 
# Now we are going to create the GPT model using the `MultiHeadSelfAttention` module. Please fill in the sections marked `TODO`.
# 
# In this cell, we are going to implement what is called a residual connection, which takes the form:
# 
# ```
# x := x + MultiHeadSelfAttention(LayerNorm(x)) + MLP(LayerNorm(x))
# ```

# In[2]:


import torch.nn as nn


class Block(nn.Module):
    """an unassuming Transformer block"""

    # === EXERCISE PART 1 START: CONSTRUCT A TRANSFORMER BLOCK ===
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # TODO: Instantiate the MultiHeadSelfAttention module
        self.attn = MultiHeadSelfAttention(config)  # Regularisation

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # TODO: implement a residual connection of the following form
        x = x + self.attn(self.ln1(x))

        # Hint: MultiHeadSelfAttention, LayerNorm, and MLP were all instantiated in __init__
        # and are available as properties of self, e.g. self.attn
        x = x + self.mlp(self.ln2(x))

        return x

    # === EXERCISE PART 1 END: CONSTRUCT A TRANSFORMER BLOCK ===


# Check that the block instantiates
block = Block(config)
block


# In[4]:


# Let's check some aspects of the block
block = Block(config)
assert isinstance(block.attn, MultiHeadSelfAttention)


# ## Step: Let's construct the GPT module
# It's time to put it all together and make our GPT PyTorch model. As before, fill in the `TODO` sections.

# In[13]:


import torch
import torch.nn.functional as F


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    # === EXERCISE PART 2 START: COMPLETE THE GPT MODEL ===

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        # TODO: Instantiate a sequence of N=config.n_layer transformer blocks.
        # Hint: use nn.Sequential to chain N instances of the Block module.
        # self.blocks = <TODO>
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print(
            "number of parameters: {}".format(sum(p.numel() for p in self.parameters()))
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # Create token embeddings and add positional embeddings
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        # TODO: Pass the embeddings through the transformer blocks, created previously in __init__
        # x = <TODO>
        x = self.blocks(x)


        # Decode the output of the transformer blocks
        x = self.ln_f(x)
        logits = self.head(x)

        # If we are given some desired targets also calculate the loss, e.g. during training
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            loss = None

        return logits, loss

    # === EXERCISE PART 2 END: COMPLETE THE GPT MODEL ===

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_tokens=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # === EXERCISE PART 3 START: COMPLETE THE GENERATION LOGIC ===
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # TODO: apply softmax to convert logits to (normalized) probabilities
            # using F.softmax. Remember the dim=-1 parameter.
            # probs = <TODO>
                probs = F.softmax(logits, dim=-1)

            # TODO: sample from the distribution (if top_k=1 this is equivalent to greedy sampling)
            # using torch.multinomial. You only need to sample a single token.
            # idx_next = <TODO>
                idx_next = torch.multinomial(probs, num_samples=1)



            # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

            # stop prediction if we produced a stop token
            if stop_tokens is not None and idx_next.item() in stop_tokens:
                return idx
        # === EXERCISE PART 3 END: COMPLETE THE GENERATION LOGIC ===

        return idx


# In[14]:


# Let's check this model runs real quickly


model = GPT(config)
input_seq = torch.tensor([[1, 2, 3]])
output_seq = model.generate(input_seq, max_new_tokens=30)

# Check the generated sequence shape
assert output_seq.shape == (1, 33)

print("input sequence:", input_seq.tolist()[0])
print("output sequence:", output_seq.tolist()[0])

print("Success!")


# ## Step: Load a dataset
# 
# We will now train our GPT model on a dataset consisting of one- to three-digit addition problems, e.g.
# 
# ```
# 111+222=3+30+300=333
# ```
# 
# We break the addition up into two steps (first `=3+30+300` and then finally `=333`) to help the model train more quickly and successfully.

# In[15]:


# Run this cell to create a dataset. No modifications are needed.

import numpy as np
from common import AdditionDataset, CharacterTokenizer

BLOCK_SIZE = 32

tokenizer = CharacterTokenizer(
    characters=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", " ", "="],
    model_max_length=BLOCK_SIZE,
)

dataset = AdditionDataset(
    tokenizer=tokenizer,
    block_size=BLOCK_SIZE,
    numbers=list(range(0, 1000, 2)),
    include_intermediate_steps=True,
)

for ix in [11177, 22222]:
    x, y = dataset[ix]

    print(f"=== Example {ix} ===")

    np.set_printoptions(linewidth=999)
    print(f"x = {x.numpy()}")
    print(f"y = {y.numpy()}")

    # show lengths
    print(f"x length = {len(x)}")
    print(f"y length = {len(y)}")

    # print x decoded
    x = tokenizer.decode(x, skip_special_tokens=True)
    print(f"x decoded = {x}")

    # print y decoded, replacing the -1 token with _
    num_unknowns = y.tolist().count(-1)
    y = tokenizer.decode(y[num_unknowns:], skip_special_tokens=True)
    print(f"y decoded = {'_'*(num_unknowns-1)}{y}")


# Let's take a minute to examine our dataset.
# 
# * What do you notice about the x, the input, and y the target?
# * Are they the same length?
# * What can you say about the alignment of the sequences?
# * What else is different between them?

# # Step: Train the model!
# 
# Now we will train a small GPT model using this dataset. Along the way we should see how the model's performance improves on real data.

# In[16]:


# Instantiate the model and a trainer. No modifications are needed.

from common import GPTConfig, Trainer, TrainerConfig

# instantiate a mini-GPT type model
model_config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=3,
    n_head=3,
    n_embd=48,
)

model = GPT(model_config)

# create a trainer
train_config = TrainerConfig(
    max_epochs=1,
    batch_size=1000,
    learning_rate=4e-3,
)

trainer = Trainer(model, dataset, train_config)

# Print the device the trainer will use (cpu, gpu, ...)
print(f"Using device: {trainer.device}")


# In[17]:


# Let's see the performance on real data before training. No modifications are needed.
from common import show_examples

show_examples(model, dataset, tokenizer, trainer.device, top_k=1, temperature=1.0)


# In[18]:


# Let's train the model for an epoch, and see the performance again.
# We will repeat this a few times to see the model improve.
# No modifications are needed.

# Note, depending on your hardware, you may need to reduce the batch size
# if you get any out-of-memory errors.

for _ in range(6):
    trainer.train()  # train just one epoch each time
    show_examples(model, dataset, tokenizer, trainer.device, top_k=3, temperature=1.0)


# In[19]:


# Let's evaluate on 30 examples. No modifications are needed.

show_examples(
    model, dataset, tokenizer, trainer.device, top_k=1, temperature=1.0, max_num=30
)


# As the above cell runs, ask yourself the following questions:
# * How quickly is the loss decreasing and what do you notice about the structure of the text generation as the loss decreases?
# * What is the train loss when you start to observe correct answers?
# * What is the train loss when you start to see all correct answers?
# * If you continue training after all answers are correct, might you see more incorrect answers?

# ## Congrats!
# 
# Congrats on training your own GPT model! Give yourself a hand! 🙌
