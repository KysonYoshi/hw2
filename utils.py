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
    Implement beam search decoding with 'beam_size' width.
    """

    # Step 1: Encode source input using the model encoder
    memory = model.encode(src, src_mask)

    # Step 2: Initialize decoder input and scores
    ys = torch.ones(beam_size, 1).fill_(start_symbol).type_as(src.data)  # Adjusted for beam size
    scores = torch.zeros(beam_size).to(src.device)  # Scores for the initial input

    # Expand memory and source mask for beam size
    memory = memory.expand(beam_size, -1, -1)
    src_mask = src_mask.expand(beam_size, -1, -1)

    # List to store sequences and scores
    sequences = ys
    sequence_scores = scores

    for i in range(max_len - 1):
        # Step 3: Decode using the model
        tgt_mask = subsequent_mask(sequences.size(1)).type_as(src.data)
        tgt_mask = tgt_mask.expand(beam_size, -1, -1)

        out = model.decode(memory, src_mask, sequences, tgt_mask)

        # Step 4: Calculate probabilities for the next token
        prob = torch.nn.functional.log_softmax(model.generator(out[:, -1]), dim=-1)  # Log-probabilities of next tokens

        # Step 5: Update scores and get top-k scores and indices
        scores_sum = sequence_scores.unsqueeze(1) + prob  # Shape: (beam_size, vocab_size)
        scores_flat = scores_sum.view(-1)  # Flatten to (beam_size * vocab_size)
        top_k_scores, top_k_indices = torch.topk(scores_flat, beam_size, dim=-1)

        # Step 6: Extract beam indices and token indices from top-k scores
        beam_indices = torch.div(top_k_indices, prob.size(-1), rounding_mode='floor')  # Indices of previous beams
        token_indices = torch.remainder(top_k_indices, prob.size(-1))  # Indices of new tokens

        # Step 7: Prepare the next decoder input
        next_decoder_input = []
        updated_scores = []
        for beam_idx, token_idx, score in zip(beam_indices, token_indices, top_k_scores):
            next_seq = torch.cat([sequences[beam_idx], token_idx.unsqueeze(0)], dim=-1)
            next_decoder_input.append(next_seq.unsqueeze(0))
            updated_scores.append(score.unsqueeze(0))

        sequences = torch.cat(next_decoder_input, dim=0)  # Update sequences with new tokens
        sequence_scores = torch.cat(updated_scores, dim=0)  # Update scores

        # Step 8: Check if all beams have ended
        finished_beams = (sequences[:, -1] == end_idx)

        if finished_beams.all():  # When all beams are finished, break
            break

    # Return the top-scored sequence
    best_sequence_index = torch.argmax(sequence_scores)
    best_sequence = sequences[best_sequence_index]

    return [best_sequence.tolist()]

        


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

