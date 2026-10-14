# 2026-10-14: NVFP4 KV cache (`--kv nvfp4`, opt-in)

`--kv nvfp4` stores the attention layers' K and V as NVFP4: after the same 256-point Hadamard rotation as `--kv q4_0`,
each 16 values get one UE4M3 scale and 4-bit E2M1 codes, in ggml's `block_nvfp4` layout (64 values in 36 bytes). A
head takes 144 bytes per cell, exactly as q4_0: 576 B per cell and layer, the same memory, the same KV streaming, MTP
ring and snapshot code. Measured on one RTX 5090 with Qwen3.8-Flash-Next IQ3_S, against q4_0 (the same memory) it is:

- **more precise:** mean KL to fp16 K/V 0.035 instead of 0.064 over four texts (better on three of them);
- **faster at reading prompts:** 1,428 instead of 1,415 tokens/s at 32K (+0.9%; int8 1,404), from a tensor-core
  attention kernel that takes 30% less time than q4_0's (within 3-8% of int8's), and a faster K/V append;
- not measurably different at decode: attention is ~4% of a decode window, NVFP4's takes 0.05 ms (0.4% of the
  window) more, and decode tokens/s did not separate the formats.

The default (`--kv int8`) and `--kv q4_0` are unchanged: their kernels are the code they were (`kv_q4_parity`,
`kv_hybrid_parity`, `kv_stream_parity`, `kv_steps_parity`, `qsa_prompt_attn_parity` pass).

## How it works

- **Quantizer (`kv_q4.cu`):** each 16 values get the UE4M3 scale of least squared error among four - ggml's
  (`quantize_row_nvfp4_ref`: amax / 6 rounded to UE4M3), one code above, two below - with each value's nearest E2M1
  code (ties to the smaller magnitude, as ggml). On Gaussian rows (what the rotation makes of K/V) the relative error
  is 0.086 instead of ggml's 0.095, the same as q4_0's 0.086 (numpy; the GPU test measures 0.0863 and 0.0953). The
  bytes are standard NVFP4: ggml's `to_float` decodes them. A 64-thread block quantizes one 64-value block and writes
  its 36 bytes as nine 32-bit words: written byte by byte, the host copy of a streamed session (PCIe) made the prompt
  path's append 7x slower than q4_0's (38 vs 5.4 ms per 8,192 cells); with wide stores it takes 2.2 ms.
- **Global scale:** the rotated values are divided by G = 1/16 before quantizing (a power of two: exact), which puts
  the UE4M3 scales in their normal range (below).
- **Prompt attention (`qsa_prompt_attn.cu`, `prompt_attn_fp4_kernel`):** the int8 kernel's structure (each warp owns
  64 dimensions and gathers its own slice of every cell with `cp.async` one chunk ahead), where a warp's slice is
  exactly one NVFP4 block: 36 bytes per cell instead of int8's 66. An NVFP4 value (a 2-significant-bit code times a
  4-bit scale, 2^-14 .. 168) is exact in FP16, so each MMA fragment is dequantized to its exact values with one half2
  multiply, and both products run as for FP16 K/V (q split into hi/lo halves, FP32 accumulation). sm_80 and newer.
- **Decode attention (`qsa_decode_attn.cu`, mode 4):** reads NVFP4 K and V straight from the pools; bit for bit the
  same output as the kernel over FP16 pools holding the same values.
- **Snapshots:** an NVFP4 K/V image is its own format (q4_0's code + 32): it never restores into a q4_0 state or the
  reverse, although the byte counts are equal (`conversation_validation_test`).

`kv_nvfp4_parity` checks all of it: the GPU's bytes against a host transcription of the quantizer (6,144 rows with
flushed, subnormal and saturated scales, signed zeros and exact ties), the round trip against ggml's quantizer, the
batched appends, the gather against ggml's `to_float`, the decode kernel against FP16 pools (bitwise), and the
tensor-core kernel against FP16 pools (8e-6 of the output scale).

## Speed

All on one RTX 5090 (32 GB, PCIe 4.0 x16), Core Ultra 9 275HX, 128 GB RAM, Windows 11, CUDA 13.4 (sm_120 build).

**Prompt attention kernel** (`qsa_prompt_attn_parity 32768 2048 20`: 2,048 queries over a 32K context with the
prompt path's selections, ms per chunk, three runs):

| KV | decode kernel | tensor-core kernel |
|---|---:|---:|
| int8 | 27.0 - 27.2 | 5.47 - 5.48 |
| q4_0 | 28.4 - 28.8 | 8.00 - 8.13 |
| **nvfp4** | 28.7 - 32.3 | **5.60 - 5.72** |
| fp16 | 26.3 | 16.6 - 16.7 |

All pass against FP64 (nvfp4 5.4e-5 against an output scale of 22.5; the FP32 decode kernel 3.2e-5).

**One 35K-token prompt, the setup's own configuration** (`--max-context 262144 --kv-resident 32768`, MTP, the
expert cache automatic; `STRATA_PREFILL_TIMING=1`, GPU time):

| KV | prompt total | projections, rotation, K/V append | attention |
|---|---:|---:|---:|
| q4_0 | 27,907 ms | 693 | 531 |
| **nvfp4** | **27,611 ms** | **494** | **371** |
| int8 | 28,096 ms | 615 | 343 |

The rest of the prompt (expert dequantization 35%, expert copies 13%, the KV staging 20%) is the same for all three.

**`tools/strata_bench.py --long`** (the same configuration; medians of three; decode medians are wide because the MTP
acceptance depends on the text):

| KV | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | 32K first token | short chat decode tok/s | agent decode tok/s |
|---|---:|---:|---:|---:|---|---|
| int8 | 705.0 | 1,358.3 | 1,403.7 | 22.89 s | 110.2 (87.7-132.5) | 140.7 (77.9-192.6) |
| q4_0 | 715.2 | 1,371.2 | 1,414.8 | 22.70 s | 109.8 (84.5-115.9) | 135.8 (80.5-181.0) |
| **nvfp4** | **716.1** | **1,381.6** | **1,427.5** | **22.49 s** | 99.6 (81.9-117.2) | 140.0 (87.4-177.7) |

The prompt ranges of three runs are within 0.3% of each median, so the prompt differences are real; the decode
ranges overlap (the MTP acceptance moves with the text each format produces).

**Decode attention** (`STRATA_VERIFY_PROFILE=1`, 32K context, 256 tokens out): 0.53 ms per window (q4_0) and 0.58 ms
(nvfp4) of a 13-14 ms window. Decode tokens/s cannot separate that: the two formats produce different text, so the MTP
acceptance differs (96 vs 113 windows for the same 256 tokens).

## The global scale (measured)

UE4M3 is normal from 2^-6 to 448. With G = 1/16, the sub-block scales of the real K/V (all 13 attention layers, 37.0
million sub-blocks each of K and V; ggml's scale rounding, before the search) used these exponent fields
(`scales-nvfp4.txt`: the count per UE4M3 code, K then V):

| | zero | subnormal | exponent field (1-15 usable) | saturated (448) |
|---|---:|---:|---|---:|
| K | 0 | 0 | 7-12 (97.6% in 9-11) | 0 |
| V | 0 | 0 | 5-12 (95% in 7-9) | 0 |

That leaves 3-4 binades on both sides (the search moves a scale by at most one code up or two down). Inside the
normal range a power-of-two G changes no result, so G = 1/16 is not a tuned value; it only keeps the scales away from
both ends. On Gaussian rows the error is flat for RMS 2^-10 .. 2^4 and grows outside it.

## Quality (teacher-forced, against fp16 K/V)

`run.py --windows`: four texts of 8-10K tokens (this repository's `docs/DETAILS.md`, `src/prefill/prefill.cpp`,
`serve/server.py`, `docs/AMD_HIP.md` as of commit 296b59d2, read with `git show`), each one message read and scored entirely through the verify windows
(`--short-read 16384`), so every format runs the same decode attention kernel in the same order and only the stored
K/V differs. One serve engine per `--kv`, `--max-context 65536`, no KV streaming, a fixed `--expert-cache 6000` (7,846
slots in every run), `--adapt-every 100000`. `STRATA_LOGPOS` with the top 20 per position; `compare.py`: KL(fp16 ||
run) over fp16's top 20 plus one bucket for the rest, and whether the most likely token is the same. Two int8 runs
gave identical files: the method is deterministic.

| text (positions) | int8 KL | q4_0 KL | nvfp4, ggml's scale | **nvfp4** | same top-1: int8 / q4_0 / nvfp4 |
|---|---:|---:|---:|---:|---|
| docs (9,686) | 0.031 | 0.161 | 0.210 | **0.072** | 93.7 / 83.1 / 88.8% |
| C++ (9,013) | 0.004 | 0.009 | 0.007 | 0.011 | 99.6 / 99.3 / 99.4% |
| Python (8,070) | 0.004 | 0.029 | 0.011 | **0.019** | 99.4 / 97.5 / 98.5% |
| docs 2 (9,968) | 0.016 | 0.058 | 0.143 | **0.040** | 96.9 / 92.6 / 94.6% |
| mean | 0.014 | 0.064 | 0.093 | **0.035** | 97.4 / 93.1 / 95.3% |

Standard errors of the KL means are 0.0002-0.004 (`compare.py` prints them); the spread between texts is larger.

- **NVFP4 beats q4_0** on three of four texts and on average (KL 45% lower), and loses on C++ (0.011 vs 0.009).
  int8 stays ahead of both.
- **The scale search is what does it:** with ggml's scale (amax / 6) NVFP4 is worse than q4_0 on average (0.093):
  after the rotation the values are close to Gaussian, where q4_0's 16 even levels lose less than E2M1's 8 magnitudes.
- **The rotation stays:** without it NVFP4 was worse on every text in the first measurement below (KL 0.085-0.224),
  and K's outlier channels pushed 0.9% of the K sub-blocks to exponent 14, six to saturation
  (`scales-nvfp4-unrotated.txt`; a temporary switch, not part of the change).
- **Why not measure through the prompt path:** `run.py` without `--windows` reads a long first message with the
  tensor-core kernels and scores a last message. There, two NVFP4 kernels that read identical bytes (the decode
  kernel and the tensor-core one) differ from each other by KL 0.03-0.05, as much as the formats differ from fp16: the
  model amplifies the summation order, as `2026-10-03-v100-prompt-attn` found. Those runs (10.5K and 35.9K-token
  prompts, ggml's scale) put NVFP4 and q4_0 within that noise.
- The perplexity of the true tokens moved by up to 0.15 nats per token even for int8 (whose KL is 0.004 on code): it
  is dominated by rare tokens outside the top 20 and says little here.

## Limits

- One model (IQ3_S), one card (RTX 5090, CUDA 13.4, sm_120), Windows 11; four texts of 8-10K tokens for quality.
- Built and tested for CUDA only. The kernels are in files the HIP build compiles too (`kv_q4.cu`,
  `qsa_decode_attn.cu`, `qsa_prompt_attn.cu`; only intrinsics those files already use, and the tensor-core kernel is
  compiled out on AMD as the others are), but **the HIP build was not run**. The SYCL (Intel) and Metal engines do not
  have the format: they reject `--kv nvfp4`.
- The tensor-core prompt kernel needs sm_80 or newer (`cp.async`); older cards read NVFP4 prompts with the decode
  kernel, as they do q4_0.
- Setup does not ask for it: `START-HERE.bat --setup --kv nvfp4` (setup refuses it for the Intel Arc and Mac
  engines), or `--kv nvfp4` in a `strata-*.json`.
- The quality runs used no KV streaming; the speed runs did (`--kv-resident 32768`).

## Reproduce

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build --target strata kv_nvfp4_parity qsa_prompt_attn_parity conversation_validation_test
./build/kv_nvfp4_parity
./build/qsa_prompt_attn_parity 32768 2048 20
python bench/results/2026-10-14-kv-nvfp4/run.py --windows <out> build/strata.exe fp16 int8 q4_0 nvfp4
python bench/results/2026-10-14-kv-nvfp4/compare.py <out> fp16 int8 q4_0 nvfp4
```

The raw log-probability files (1-3 MB each) are not in the repository; `run.py` writes them again.
