# Informer forecasting

This page documents KairoTS's paper-first Informer architecture and its packed history/calendar input adapter.

## Sources

- Zhou et al., [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436), AAAI 2021, arXiv v3.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), cited for sinusoidal positions, head concatenation and feed-forward operations.
- The source conflicts and selected conventions are recorded in [`explanation/notes.md`](../../explanation/notes.md#informer-source-choices).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Value, positional and calendar embeddings | `models/Informer/DataEmbedding.py` | Circular scalar convolution; window-local zero-based sinusoidal positions; categorical month/day/hour and optional quarter-hour embeddings | `test_informer_embeddings` |
| ProbSparse attention and context initialization (Eqs. 3–4, Algorithm 1) | `layers/Attention.py` | Sample `factor × ceil(log L)` keys per query; select top sparse queries; unsampled score entries are zero; decoder self-attention uses causal cumulative-sum context | `test_informer_sparse_attention` |
| Distilled three-layer encoder plus independent one-layer replica | `models/Informer/Encoder.py`, `Distillation.py` | Two length-halving conv/max-pool stages on the main path; replica reads the last quarter of the original embedded history; concatenate both encoded paths | `test_informer_encoder_stacks` |
| Two-layer decoder with masked self-attention, full cross-attention and positionwise feed-forward | `models/Informer/DecoderLayer.py`, `FeedForward.py` | Eight decoder heads by default; cross-attention is dense | `test_informer_dense_oracle`, `test_informer_paper_defaults` |
| Future-value placeholders with known future timestamps | `models/Informer/__init__.py`, `tasks/Forecasting.py` | Decoder input is the last `label_length` history values plus zero future values; future calendar marks remain visible | `test_informer_future_values`, `test_informer_future_calendar` |

## Initialization and protocol

The defaults use three full-history encoder layers, a separate one-layer last-quarter replica, two decoder layers, `d_model=512`, `d_ff=2048`, 16 encoder heads, 8 decoder heads and sampling factor 5.
Omitted projection, initialization, calendar-vocabulary and normalization details use the pinned original author implementation.
Sparse query sampling remains stochastic at evaluation.

The printed positional formula has an index range that extends beyond its embedding width.
KairoTS uses the cited Transformer convention and original author encoding: positions restart at zero for each window, use base 10,000, and alternate sine/cosine channels.
This source decision is explicit in the notes.

`configs/forecasting/ETTh1-Informer.json` uses history and horizon 96, batch size 32, Adam at `1e-4`, ten one-epoch centralized updates, and the paper model on the released ETTh1 12/4/4-month borders.
Its controlled comparison with TSLib is an implementation diagnostic, not a claim of reproducing the paper's reported metric: TSLib uses a different encoder/decoder depth, ProbSparse cross-attention, and continuous calendar projection.
