# PRISM

**Local LLM Instrumentation, Tracing, and Replay Platform**

PRISM hooks non-invasively into [llama.cpp](https://github.com/ggerganov/llama.cpp)'s inference pipeline via the `ggml_backend_sched_eval_callback` mechanism — no model source changes required. It captures real-time tensor metadata (shape, dtype, activation stats, latency) as tokens flow through the architecture and presents everything in an interactive terminal UI.

```
 PRISM v0.3   [Tab]:Cycle Focus  [Q]:Quit  [j/k]:Navigate  [h/l]:Pan  [+/-]:Contrast
╭█ 1. MODEL TOPOLOGY──────────────╮╭  2. LIVE PACKET STREAM───────────────────────────╮
│▼ llama-3-8b (demo)              ││  ID  │ TIMESTAMP    │ LAYER TYPE    │ DEVICE      │
│  ● token_embd                   ││──────┼──────────────┼───────────────┼─────────────│
│  ▼ layers                       ││  100 │ 14:22:01.33  │ Embedding     │ CPU         │
│    ▶ blk.0                      ││  101 │ 14:22:01.33  │ RMSNorm       │ CPU         │
│    ▼ blk.1  ← [capture target]  ││  102 │ 14:22:01.33  │ Attn (Self)   │ Metal       │
│      ● blk.1.attn_norm          ││  103 │ 14:22:01.33  │ MLP (SwiGLU)  │ Metal       │
│      ● blk.1.attn               ││  104 │ 14:22:01.34  │ Attn (Self)   │ Metal       │
│    ▶ blk.2                      ││  105 │ 14:22:01.34  │ RMSNorm       │ CPU         │
╰─────────── [j/k] Navigate ──────╯╰──────────────────────────────────────────────────╯
╭  3. ATTENTION MATRIX (HEAD 0)────────────────────────────────────────────────────────╮
│          [I]    [want]  [it]    [to]    [be]    [kbrd]                               │
│ [I]      ██      ··      ··      ··      ··      ··                                  │
│ [want]   ▓▓      ██      ··      ··      ··      ··                                  │
│ [it]     ░░      ▒▒      ██      ··      ··      ··                                  │
│ [to]     ░░      ░░      ▒▒      ██      ··      ··                                  │
│ HEAD 0/3   [</>]:Head  [hjkl]:Pan  [+/-]:Contrast ×1.0    Viewport [0-7] × [0-7]   │
╰──────────────────────────────────────────────────────────────────────────────────────╯
╭  4. RUNTIME METRICS──────────────╮╭  5. ANOMALY LEDGER────────────────────────────╮
│ Layer  : blk.1.attn              ││ ⚠ 14:22:01.34  Outlier blk.1.attn: Max=6.5>6  │
│ Type   : Attn (Self)             ││ ✓ no further anomalies                         │
│ Shape  : [1, 32, 4096]           ││                                                │
│ DType  : float16                 ││                                                │
│ Mean   : 0.0312   Max : 6.5000   ││                                                │
│ Sparse : ████████████░░  49%     ││                                                │
│ Latency: 1.190 ms                ││                                                │
╰──────────────────────────────────╯╰────────────────────────────────────────────────╯
```

---

## Features

- **Non-invasive hook** — registers `ggml_eval_cb` in `llama_context_params`; zero changes to model code
- **Per-tensor telemetry** — layer name, type, compute device, latency (μs), shape, dtype, mean/max/min, sparsity
- **Anomaly detection** — flags activations where `|max| > 6.0` or unexpected CPU fallbacks
- **Attention matrix visualiser** — block-character heatmap with pan, head cycling, and contrast control
- **Fixed-size ring buffer** — stores up to 256 packets; oldest entry silently overwritten when full, keeping RAM bounded
- **Trace save / replay** — export a capture to a human-readable `.prism` file; replay later without a model
- **Interactive TUI** — five panels, vim-style keybindings, 10 Hz live refresh

---

## Requirements

| Dependency | Fetched automatically via CMake `FetchContent` |
|---|---|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Inference backend + Metal/BLAS backends |
| [FTXUI v5](https://github.com/ArthurSonzogni/FTXUI) | Terminal UI framework |

**Toolchain:** C++17, CMake ≥ 3.16, Apple Clang / GCC / MSVC

**macOS:** Metal GPU backend is detected and enabled automatically.  
**Linux:** BLAS / CUDA backends follow the same llama.cpp CMake flags.

---

## Build

```bash
git clone https://github.com/<your-handle>/prism.git
cd prism

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel         # first build downloads + compiles llama.cpp (~3 min)

./build/prism --help
```

> **Tip:** subsequent builds are incremental and take only a few seconds.

---

## Usage

### Demo mode (no model required)
```bash
./build/prism
```
Seeds the TUI with synthetic telemetry so you can explore all five panels immediately.

### Live inference
```bash
./build/prism path/to/model.gguf "The quick brown fox"
```
Runs a single forward pass with the hook active. Captured packets appear in the TUI in real time.

Download any GGUF model from [HuggingFace](https://huggingface.co/models?library=gguf), for example:
```bash
# ~2 GB, runs on CPU or Apple Silicon
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
./build/prism llama-2-7b.Q4_K_M.gguf "Explain transformers in one sentence"
```

### Save a trace
```bash
./build/prism model.gguf "my prompt" --save capture.prism
```
Saves every captured packet and anomaly to `capture.prism` after inference completes.

### Replay a trace
```bash
./build/prism --replay capture.prism
```
Loads the trace into the TUI — no model file needed. Useful for sharing captures or post-hoc analysis.

---

## TUI Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle focus forward / backward between panels |
| `j` / `k` | Navigate topology tree / scroll packet stream |
| `Space` | Expand or collapse a topology node |
| `h` `j` `k` `l` | Pan the attention matrix |
| `<` / `>` | Previous / next attention head |
| `+` / `-` | Increase / decrease attention contrast |
| `Q` | Quit |

---

## Trace File Format

`.prism` files are plain text and human-readable:

```
PRISM_TRACE v1
META model=llama-3-8b
META saved_at=1782814202027848

PKT 100 1782814202027379 1 2 1.142 | blk.0.attn | [1,32,4096] | float16 | 0.031 3.2 -2.9 0.54
PKT 101 1782814202027415 2 2 0.871 | blk.0.ffn_out | [1,32,4096] | float16 | 0.022 5.1 -4.8 0.61
ANO 1782814202027415 0 | Outlier Feature blk.1.attn: Max = 6.5 > 6.0
```

Fields: `PKT <id> <timestamp_µs> <layer_type> <device> <latency_ms> | <name> | <shape> | <dtype> | <mean> <max> <min> <sparsity>`

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    llama.cpp inference                  │
│   llama_context_params.cb_eval = ModelHook::ggml_eval_cb│
└───────────────────────┬─────────────────────────────────┘
                        │ ask=true  (before tensor)
                        │ ask=false (after tensor, data valid)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                      ModelHook                          │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │ should_capture()│  │ on_tensor_after()            │  │
│  │ filter by name  │  │  measure latency             │  │
│  │ pattern         │  │  compute stats (CPU tensors) │  │
│  └────────┬────────┘  │  detect anomalies            │  │
│           │           │  capture KQ_soft_max → attn  │  │
│           └───────────┴──────────────┬───────────────┘  │
│                                      │                  │
│         RingBuffer<TelemetryPacket, 256>                │
│         RingBuffer<AnomalyRecord,    64>                │
└──────────────────────────────────────┬──────────────────┘
                                       │ thread-safe reads
                        ┌──────────────▼──────────────┐
                        │           TuiApp             │
                        │  Panel 1: Model Topology     │
                        │  Panel 2: Live Packet Stream │
                        │  Panel 3: Attention Matrix   │
                        │  Panel 4: Runtime Metrics    │
                        │  Panel 5: Anomaly Ledger     │
                        └──────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │       prism_trace::save()   │
                        │       prism_trace::load()   │
                        │       .prism trace file     │
                        └─────────────────────────────┘
```

---

## Project Structure

```
prism/
├── CMakeLists.txt
├── include/
│   ├── core/
│   │   ├── ring_buffer.h     Thread-safe fixed-capacity ring buffer
│   │   ├── telemetry.h       TelemetryPacket, AnomalyRecord, ModelNode
│   │   ├── hook.h            ModelHook — ggml eval callback interface
│   │   └── trace_io.h        Trace save / load declarations
│   └── tui/
│       └── app.h             TuiApp — FTXUI-based interactive interface
└── src/
    ├── main.cpp              Entry point, argument parsing, inference thread
    ├── core/
    │   ├── hook.cpp          Capture filter, stat computation, anomaly detection
    │   └── trace_io.cpp      .prism serialisation / deserialisation
    └── tui/
        └── app.cpp           Five-panel TUI renderer and event handler
```
