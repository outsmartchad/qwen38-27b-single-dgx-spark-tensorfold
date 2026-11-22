# How Qwen3.8 27B works with TensorFold

## How the model works

This recipe serves the pinned `Vontra/Qwen3.8-27B-MLX-4bit` target at `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` and `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` as the draft model. Qwen3.8-27B is a 27B dense vision-language model with 64 layers and a 5120-wide hidden state. The target uses MLX 4-bit weights, while TensorFold stores KV and prefill activations in FP8. [official model card](https://huggingface.co/Qwen/Qwen3.8-27B) · [target snapshot](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c) · [draft snapshot](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4)

Each four-layer group has three Gated DeltaNet layers followed by one Gated Attention layer, with a feed-forward network after each mixer. GDN uses 48 value heads and 16 query-key heads with 128-wide heads. The full-attention layer uses 24 query heads and 4 KV heads with 256-wide heads and 64 rotary dimensions. GDN carries a fixed-size recurrent state, so most layers avoid growing KV storage, while the periodic full-attention layer preserves direct long-range retrieval. [official model card](https://huggingface.co/Qwen/Qwen3.8-27B) · [TensorFold repository](https://github.com/ashhart/TensorFold)

DFlash2 predicts candidates for several positions with a parallel diffusion drafter. A compact selector scores adjacent candidates for coherence, then the target verifies the resulting block. Accepted proposals let one target pass commit several tokens. Standard exact verification preserves the target distribution. This profile uses FP8 KV and FP8 prefill activations, so long prompts and concurrent slots draw from the same unified-memory budget. [DFlash paper](https://arxiv.org/abs/2602.06036) · [DFlash 2 design](https://inco.ai/blog/dflash2/) · [Mia pinned recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold/tree/c11ff138871cf43d12898d79dbfa4dc952d17ca4)

## How this recipe runs it on one DGX Spark

```text
target + DFlash2 snapshots -> TensorFold v0.6.0
                            -> prefill and FP8 KV pool
                            -> DFlash2 proposal / target verify
                            -> localhost streaming response
```

`serve.sh download` pulls the pinned image and snapshots. `start` launches TensorFold on `127.0.0.1:8888` with eight slots, a 262,144-token context, a 32 GiB KV pool, FP8 KV, and FP8 prefill activations. The measured run set `TENSORFOLD_MEMORY_RESERVE_GIB=0`; the launcher default is 8 GiB, so reproducing the numbers requires an isolated benchmark window with the same reserve. [Launcher](../serve.sh) · [runtime notes](../README.md#runtime)

The decode sweep used one through five concurrent streams, 400 output tokens, thinking enabled, temperature 0.7, top-p 1, and top-k 20. Best aggregate rates were 43.8, 79.9, 105.4, 121.5, and 140.2 tokens per second. Separate cold-prefill runs reached 1,740 tokens per second at 32,568 input tokens and 963 tokens per second at 199,061 input tokens, with exact retrieval. [Raw benchmark](../results/benchmark-2026-11-20.json)

Eight slots create room for batching, while the five-stream sweep shows the tested contention point. The 32 GiB KV pool and zero-reserve measurement maximize capacity on this isolated host and leave less protection for the operating system. The test scope covers localhost decode, two fresh retrieval prompts, and model discovery, with media inputs and network serving outside this run. [README benchmark scope](../README.md#benchmarks)
