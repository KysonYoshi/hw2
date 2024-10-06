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
    Beam search decoding with 'beam_size' width.

    Args:
        model: The trained model to be used for decoding.
        src: The source sentence (input) tensor.
        src_mask: The source mask for padding.
        max_len: The maximum length of the decoded sentence.
        start_symbol: The index of the start symbol (<sos>) for the decoder.
        beam_size: The number of beams (hypotheses) to maintain during search.
        end_idx: The index of the end symbol (<eos>).

    Returns:
        The best decoded sequence as a list of token indices.
    """

    # Encode the source input using the encoder
    memory = model.encode(src, src_mask)

    # Initialize the decoder input with the start symbol and set beam width
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # (1, 1) shape, start with <sos>
    scores = torch.zeros(1).cuda()  # Initialize with zero scores for the initial beam

    # Prepare for beam search, maintaining beams in each step
    finished_beams = []
    for i in range(max_len - 1):
        # Decode step: model.decode() to get the next output from the decoder
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))

        # Get the last output (the token scores for the current step)
        out = out[:, -1, :]  # (beam_size, vocab_size)
        prob = torch.nn.functional.log_softmax(out, dim=-1)  # Log probabilities for stability

        # If the first step, expand to beam_size (handle initial beam)
        if i == 0:
            scores, indices = prob[0].topk(beam_size)  # Select top-k beams at the first step
            ys = ys.expand(beam_size, 1)  # Expand ys for beam size
        else:
            scores = scores.unsqueeze(1) + prob  # Expand scores and add log probabilities
            scores, indices = scores.view(-1).topk(beam_size)  # Get top-k scores and their indices

        # Extract beam and token indices from top-k scores
        beam_indices = indices // prob.size(-1)  # beam index (from expanded scores)
        token_indices = indices % prob.size(-1)  # token index (actual vocab token)

        # Prepare next decoder input, handle <eos> cases
        next_ys = []
        for beam_idx, token_idx in zip(beam_indices, token_indices):
            if token_idx == end_idx:
                finished_beams.append((ys[beam_idx].clone(), scores[beam_idx].item()))
            else:
                next_ys.append(torch.cat([ys[beam_idx], token_idx.unsqueeze(0)]))

        if len(finished_beams) == beam_size:
            break

        # Update ys for the next iteration with the best beams
        ys = torch.stack(next_ys)  # Stack the best beams as the new ys

        # If all beams are completed (i.e., all end tokens found), we can stop early
        if len(next_ys) == 0:
            break

        # Expand encoder output for beam size (only at the first step)
        if i == 0:
            memory = memory.expand(beam_size, *memory.shape[1:])

    # If some beams reached <eos>, pick the one with the highest score, otherwise, return the longest beam
    if len(finished_beams) > 0:
        best_beam = max(finished_beams, key=lambda x: x[1])[0]
    else:
        best_beam = ys[0]

    return [best_beam.tolist()]
        


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

