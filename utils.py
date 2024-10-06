import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu


## Dummy functions defined to use the same function run_epoch() during eval
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    """
    Implement beam search decoding with 'beam_size' width
    """
    # Step 1: Encode source input using the model
    memory = model.encode(src, src_mask)

    # Initialize decoder input with the start symbol and set scores to 0
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # Initial decoder input, shape (1, 1)
    scores = torch.zeros(1, beam_size).cuda()  # Initialize scores for each beam

    # Expand memory and src_mask for the beam size
    memory = memory.expand(beam_size, -1, -1)
    src_mask = src_mask.expand(beam_size, -1, -1)

    completed_sequences = []
    completed_scores = []

    for i in range(max_len - 1):
        # Step 2: Decode using the model
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))

        # Step 3: Calculate probabilities for the next token
        prob = F.log_softmax(model.generator(out[:, -1]), dim=-1)  # Get the log probabilities for the next token

        # If the end token has been generated, set its probability to -inf for any subsequent positions
        prob[:, end_idx] = prob[:, end_idx].masked_fill(ys[:, -1] == end_idx, 0)

        # Step 4: Update scores by adding the new probabilities
        if i == 0:
            scores = prob[0]  # Initialize scores for the first time
        else:
            scores = (scores.unsqueeze(1) + prob).view(-1)  # Add new token probabilities to current scores

        # Step 5: Get top-k scores and their corresponding token indices
        top_k_scores, top_k_indices = scores.topk(beam_size)

        # Step 6: Extract beam indices and token indices from top-k scores
        beam_indices = torch.div(top_k_indices, prob.size(1), rounding_mode='floor')  # Beam indices
        token_indices = top_k_indices % prob.size(1)  # Token indices

        # Prepare next decoder input by adding the selected token to each beam
        next_decoder_input = []
        for beam_idx, token_idx in zip(beam_indices, token_indices):
            # If the beam has already generated the end token, keep it in the completed sequences
            if ys[beam_idx][-1] == end_idx:
                completed_sequences.append(ys[beam_idx])
                completed_scores.append(top_k_scores[beam_idx])
            else:
                next_decoder_input.append(torch.cat([ys[beam_idx], token_idx.view(1, 1)], dim=1))

        if len(next_decoder_input) == 0:
            # All beams have completed
            break

        # Update ys with the new decoder input for the next iteration
        ys = torch.cat(next_decoder_input, dim=0)

        # Check if all beams are finished, exit if true
        if len(completed_sequences) >= beam_size:
            break

    # If any sequences remain unfinished, add them to the completed sequences
    if len(completed_sequences) < beam_size:
        for beam_idx in range(beam_size):
            if ys[beam_idx][-1] != end_idx:
                completed_sequences.append(ys[beam_idx])
                completed_scores.append(top_k_scores[beam_idx])

    # Step 7: Return the top-scored sequence
    # Find the best sequence with the highest score
    best_sequence_idx = torch.argmax(torch.stack(completed_scores))
    best_sequence = completed_sequences[best_sequence_idx]

    # Convert the best sequence to a list of token indices
    return best_sequence.tolist()
        


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for s in batch:
        _src = s['de']
        _tgt = s['en']
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def remove_start_end_tokens(sent):

    if sent.startswith('<s>'):
        sent = sent[3:]

    if sent.endswith('</s>'):
        sent = sent[:-4]

    return sent


def compute_corpus_level_bleu(refs, hyps):

    refs = [remove_start_end_tokens(sent) for sent in refs]
    hyps = [remove_start_end_tokens(sent) for sent in hyps]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return bleu.score

