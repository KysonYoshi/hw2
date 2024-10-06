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
        model: The trained transformer model.
        src: Source input tensor (batch size, seq length).
        src_mask: Mask for source input.
        max_len: Maximum length of the output sequence.
        start_symbol: Index of the start token.
        beam_size: Number of beams to keep.
        end_idx: Index of the end token.

    Returns:
        Decoded sequence of tokens.
    """

    # Step 1: Encode the source input using the model
    memory = model.encode(src, src_mask)

    # Step 2: Initialize the beam with the start token and scores
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # Initial input is start token
    scores = torch.zeros(1, 1).cuda()  # Scores initialized with 0

    finished_beams = []  # Store the completed sequences
    beams = [(ys, scores)]  # Start with the initial beam containing the start token

    for i in range(max_len - 1):
        all_candidates = []  # Store all beam candidates

        # For each beam, expand it with the next token
        for ys, scores in beams:
            # Step 3: Decode using the model
            out = model.decode(ys, memory, src_mask, subsequent_mask(ys.size(1)).cuda())
            out = out[:, -1]  # Get the output of the last step
            prob = torch.nn.functional.log_softmax(out, dim=-1)  # Apply log-softmax to get probabilities

            # Step 4: Update the scores by adding the log probabilities to the current beam scores
            next_scores = scores + prob.squeeze(0)

            # Get top-k scores and indices (for beam_size * vocab_size candidates)
            topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)

            # Step 5: Extract beam indices and token indices from top-k scores
            beam_indices = topk_indices // model.vocab_size
            token_indices = topk_indices % model.vocab_size

            # Step 6: Create new beam candidates
            for k in range(beam_size):
                token = token_indices[k].item()
                new_ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(token)], dim=1)
                new_scores = topk_scores[k].view(1, 1)

                # If the token is the end token, add this beam to finished_beams
                if token == end_idx:
                    finished_beams.append((new_ys, new_scores))
                else:
                    all_candidates.append((new_ys, new_scores))

        # Step 7: Select top-k beams for the next step
        beams = sorted(all_candidates, key=lambda x: x[1].item(), reverse=True)[:beam_size]

        # Check if all beams have finished
        if len(finished_beams) == beam_size:
            break

    # Step 8: Return the top-scoring finished beam
    if len(finished_beams) > 0:
        top_beam = max(finished_beams, key=lambda x: x[1].item())
        return top_beam[0].squeeze(0).tolist()
    else:
        top_beam = beams[0]
        return top_beam[0].squeeze(0).tolist()
        


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

