# Qwen3.8 27B on one DGX Spark

TensorFold v0.6.0 runs the Vontra MLX 4-bit checkpoint and z-lab DFlash2 draft on one NVIDIA GB10. The TensorFold API binds to `127.0.0.1:8888`.

[How the model works](docs/how-it-works.md) · [Quick start](#quick-start) · [Runtime](#runtime) · [Raw results](results/benchmark-2026-11-20.json)

## Benchmarks

Measured November 20, 2026 on one DGX Spark in benchmark mode. The benchmark sent direct HTTP requests to TensorFold. The Flash Next serving container was paused during the run.

### Decode, prose

SparkDash hash-map prompt with thinking on, temperature 0.7, top-p 1, top-k 20 and 400 generated tokens per request. Each row uses the wave with the highest aggregate decode rate across three runs. Per-request decode and TTFT are medians from that same wave. TTFT counts the first streamed reasoning or answer token.

| Streams | Best aggregate tok/s | Per request tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 43.8 | 43.8 | 98 ms |
| 2 | 79.9 | 41.8 | 185 ms |
| 3 | 105.4 | 37.8 | 281 ms |
| 4 | 121.5 | 33.6 | 389 ms |
| 5 | 140.2 | 29.8 | 496 ms |

### Cold prefill

Fresh prompts with zero cached prompt tokens reported and thinking off. The input rate divides actual prompt tokens by client TTFT. The runner sized prompts with a separate calibration request because TensorFold v0.6.0 did not expose a tokenizer endpoint.

| Prompt tokens | Input tok/s to first token | TTFT | Retrieval |
| ---: | ---: | ---: | --- |
| 32,568 | 1,740 | 18.72 s | Exact key |
| 199,061 | 963 | 206.78 s | Exact key |

## Runtime

- TensorFold v0.6.0 on `linux/arm64`, pinned to the exact image digest in [serve.sh](serve.sh).
- [Vontra MLX 4-bit target](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c) with [z-lab DFlash2 draft](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4). The container log reports drafts on.
- FP8 KV and FP8 prefill activations. The TensorFold KV pool setting is 32 GiB. The server has eight slots and a configured 262,144-token context. This benchmark exercised up to five simultaneous short decode requests.
- The monitor sampled a 49,567 MiB peak GPU footprint during the separate long-prefill phase. Docker reported no fixed container memory limit.
- The measured run used `TENSORFOLD_MEMORY_RESERVE_GIB=0`. The launcher defaults to an 8 GiB reserve. Set the variable to 0 when reproducing the measured profile in an isolated benchmark window.

## Quick start

Use a single DGX Spark with Docker, NVIDIA Container Toolkit, Python 3, curl and the Hugging Face CLI. The weights and draft need about 20 GB of disk space plus the container image and runtime cache. Stop other model servers before launching TensorFold.

```bash
./serve.sh download
./serve.sh start
docker logs -f spark-recipe-qwen38-27b-tensorfold
```

Once the localhost API is ready, check the served model and run the direct benchmark on the same Spark. The second argument to the prefill runs chooses the synthetic prompt variant.

```bash
curl -fsS http://127.0.0.1:8888/v1/models
export SPARK_BENCHMARK_URL=http://127.0.0.1:8888
export SPARK_BENCHMARK_MODEL=Qwen3.8-27B
python3 benchmark.py > decode-local.json
python3 benchmark.py 32768 1 > prefill-32k-local.json
python3 benchmark.py 200000 2 > prefill-200k-local.json
./serve.sh stop
```

The launcher pins both Hugging Face snapshots in its dedicated cache and starts the server with offline mode. This recipe builds on [Mia AI Lab’s pinned TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold/tree/c11ff138871cf43d12898d79dbfa4dc952d17ca4). Our scripts, documentation and synthetic fixtures use the [MIT license](LICENSE). Upstream images and weights carry their own licenses.
