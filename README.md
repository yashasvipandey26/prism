# PRISM 🔬

> **Local LLM Instrumentation, Tracing, and Replay Platform**

Ever wondered *what actually happens inside a language model* when it processes text? Which layers are slow? Where do the numbers explode? What does the attention pattern for a given sentence look like?

PRISM answers those questions — live, in your terminal, without touching a single line of model code.

---

## What is this?

When a language model runs, it's really just a long chain of mathematical operations — embeddings, attention, feed-forward networks, normalization — stacked dozens of times. These operations are normally a black box.

PRISM **taps into that chain** and records what's happening at every layer: how long each step takes, what the numbers look like, whether anything looks suspicious. It then shows all of this in a live, interactive terminal dashboard.

Think of it like **Chrome DevTools, but for a transformer model running on your machine.**

---

## What does it show?

```
 PRISM v0.3  [Tab]:Cycle Focus  [Q]:Quit                    Focus: TOPOLOGY  Packets: 12
╭█ 1. MODEL TOPOLOGY──────────────╮╭  2. LIVE PACKET STREAM──────────────────────────────╮
│▼ llama-3-8b                     ││  ID  │ TIMESTAMP    │ LAYER TYPE    │ DEVICE         │
│  ● token_embd                   ││──────┼──────────────┼───────────────┼────────────────│
│  ▼ layers                       ││  100 │ 14:22:01.33  │ Embedding     │ CPU            │
│    ▶ blk.0                      ││  101 │ 14:22:01.33  │ RMSNorm       │ CPU            │
│    ▼ blk.1                      ││  102 │ 14:22:01.33  │ Attn (Self)   │ Metal          │
│      ● blk.1.attn_norm          ││  103 │ 14:22:01.34  │ MLP (SwiGLU)  │ Metal          │
│      ● blk.1.attn  ◀ captured   ││  104 │ 14:22:01.34  │ Attn (Self)   │ Metal          │
│    ▶ blk.2                      ││  105 │ 14:22:01.34  │ RMSNorm       │ CPU            │
╰────────── [j/k] Navigate ───────╯╰─────────────────────────────────────────────────────╯
╭  3. ATTENTION MATRIX (HEAD 0)────────────────────────────────────────────────────────────╮
│           [I]     [want]   [it]    [to]    [be]    [kbrd]                                │
│  [I]      ██       ··       ··      ··      ··       ··                                  │
│  [want]   ▓▓       ██       ··      ··      ··       ··                                  │
│  [it]     ░░       ▒▒       ██      ··      ··       ··                                  │
│  [to]     ░░       ░░       ▒▒      ██      ··       ··                                  │
│  HEAD 0/3    [</>]:Head   [hjkl]:Pan   [+/-]:Contrast ×1.0      Viewport [0-7]×[0-7]   │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
╭  4. RUNTIME METRICS──────────────╮╭  5. ANOMALY LEDGER──────────────────────────────────╮
│ Layer  : blk.1.attn              ││ ⚠ 14:22:01.34  Outlier blk.1.attn: Max=6.5 > 6.0   │
│ Shape  : [1, 32, 4096]           ││ ✓ no further anomalies detected                     │
│ DType  : float16                 ││                                                     │
│ Mean   : 0.0312   Max : 6.5000   ││                                                     │
│ Sparse : ████████████░░  49%     ││                                                     │
│ Latency: 1.190 ms                ││                                                     │
╰──────────────────────────────────╯╰─────────────────────────────────────────────────────╯
```

---

## Five panels, one glance

| Panel | What it tells you |
|---|---|
| **Model Topology** | The full layer tree of the model — expand/collapse blocks, navigate with `j/k` |
| **Live Packet Stream** | Every tensor that was intercepted, timestamped, with its layer type and device |
| **Attention Matrix** | Heatmap of which tokens attend to which — scroll across heads, pan, adjust contrast |
| **Runtime Metrics** | Shape, data type, mean/max/min, sparsity bar, and latency for the selected layer |
| **Anomaly Ledger** | Auto-flagged warnings: activation values that are unusually large, unexpected CPU fallbacks |

---

## How does it work?

llama.cpp (the C++ inference engine this is built on) provides a hook called `cb_eval` — a callback function it calls **before and after computing every tensor** in the model's computation graph. We register our own function there.

That's the entire trick. No model modification. No wrapping. No forking. Just:

```
model runs → our callback fires for each tensor → we record what we see → TUI displays it
```

The callback measures the time between "before" and "after" for each tensor (that's the latency), reads the tensor's shape and data type, computes mean/max/min/sparsity on the raw numbers, and pushes everything into a **ring buffer** that holds the last 256 captures. The TUI reads from that ring buffer at 10 times per second and redraws.

For the attention matrix, we intercept the specific tensor named `KQ_soft_max` — this is the softmax output of the query×key product, which is exactly the attention weight matrix that tells you "how much does token A look at token B."

---

## Features

- **Zero model modification** — hooks in via a single callback registration, works with any GGUF model
- **Live latency profiling** — see which layers dominate compute time as inference runs
- **Attention visualization** — block-character heatmap with per-head navigation and contrast control
- **Sparsity tracking** — see what fraction of activations are near-zero (useful for understanding model behavior)
- **Anomaly detection** — automatically flags activation values above a threshold or layers that fall back to CPU unexpectedly  
- **Bounded memory** — ring buffer caps RAM usage regardless of how long inference runs
- **Trace save/replay** — export a full capture to a `.prism` file; replay it in the TUI later without needing the model
- **Works without a model** — demo mode seeds the TUI with synthetic data so you can explore immediately

---

## Build

**Requirements:** C++17, CMake ≥ 3.16, a C compiler. Everything else is downloaded automatically.

```bash
git clone https://github.com/<your-handle>/prism.git
cd prism

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The first build downloads and compiles llama.cpp and FTXUI — expect about 3–5 minutes. Subsequent builds are incremental and take seconds.

On **macOS Apple Silicon**, Metal GPU acceleration is detected and enabled automatically.  
On **Linux**, CPU + BLAS backends are used by default; CUDA can be enabled via llama.cpp's standard cmake flags.

---

## Running

### Just explore (no model needed)

```bash
./build/prism
```

Opens the TUI with synthetic data so you can try all the controls right away.

---

### With a real model

Download any GGUF file. A good small starting point (~2 GB, runs on CPU):

```bash
# using huggingface-cli (use the command: pip install huggingface_hub)
hf download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir .
```

Then run:

```bash
./build/prism llama-2-7b.Q4_K_M.gguf "Explain attention in transformers"
```

The model runs a single forward pass. Every tensor fired during that pass is captured and shown in the TUI.

---

### Save a trace

```bash
./build/prism model.gguf "my prompt" --save my_run.prism
```

After inference finishes, the full capture is written to `my_run.prism`.

---

### Replay a saved trace

```bash
./build/prism --replay my_run.prism
```

Loads the trace and opens the TUI — **no model file needed**. Useful for sharing captures with teammates or reviewing a run later.

---

## Keybindings

| Key | What it does |
|-----|-------------|
| `Tab` | Move focus to the next panel |
| `Shift+Tab` | Move focus to the previous panel |
| `j` / `k` | Move cursor down/up in topology; scroll packet stream |
| `Space` | Expand or collapse a node in the topology tree |
| `h` `j` `k` `l` | Pan the attention matrix left/down/up/right |
| `<` / `>` | Switch to previous/next attention head |
| `+` / `-` | Increase/decrease attention contrast |
| `Q` | Quit |

---

## Trace file format

`.prism` files are plain text — you can open them in any editor:

```
PRISM_TRACE v1
META model=llama-2-7b
META saved_at=1782814202027848

PKT 100 1782814202027379 1 2 1.142 | blk.0.attn | [1,32,4096] | float16 | 0.031 3.2 -2.9 0.54
PKT 101 1782814202027415 2 2 0.871 | blk.0.ffn_out | [1,32,4096] | float16 | 0.022 5.1 -4.8 0.61
ANO 1782814202027415 0 | Outlier Feature blk.1.attn: Max = 6.5 > 6.0
```

Each `PKT` line is one tensor capture: id, timestamp, layer type, device, latency, name, shape, dtype, and activation stats. Each `ANO` line is a flagged anomaly.

---

## Tech stack

| Component | Technology |
|---|---|
| Inference engine | [llama.cpp](https://github.com/ggerganov/llama.cpp) — runs GGUF models on CPU/Metal/CUDA |
| Hook mechanism | `ggml_backend_sched_eval_callback` in `llama_context_params` |
| Terminal UI | [FTXUI v5](https://github.com/ArthurSonzogni/FTXUI) — declarative C++ TUI framework |
| Ring buffer | Custom lock-free-read, mutex-write template (`RingBuffer<T, N>`) |
| Language | C++17 |
| Build system | CMake with FetchContent for dependencies |
