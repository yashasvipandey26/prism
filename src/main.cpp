#include "core/hook.h"
#include "core/trace_io.h"
#include "tui/app.h"

#include <llama.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ── Usage ─────────────────────────────────────────────────────────────────────

static void print_usage(const char* argv0) {
    std::cout <<
        "PRISM - LLM Instrumentation, Tracing, and Replay Platform\n\n"
        "Usage:\n"
        "  " << argv0 << "                              Demo mode (synthetic data)\n"
        "  " << argv0 << " <model.gguf> [prompt]        Live inference + TUI\n"
        "  " << argv0 << " <model.gguf> [prompt] --save <out.prism>  Save trace\n"
        "  " << argv0 << " --replay <trace.prism>        Replay saved trace\n\n"
        "TUI keybindings:\n"
        "  [Tab]        Cycle focus between panels\n"
        "  [j/k]        Navigate topology / scroll stream\n"
        "  [Space]      Expand / collapse topology node\n"
        "  [h/j/k/l]    Pan attention matrix\n"
        "  [</> ]       Previous / next attention head\n"
        "  [+/-]        Adjust attention contrast\n"
        "  [Q]          Quit\n";
}

// ── Synthetic seed (no model needed) ─────────────────────────────────────────

static void seed_synthetic(ModelHook& hook) {
    ModelNode root;
    root.name     = "llama-3-8b (demo)";
    root.expanded = true;

    auto leaf = [](const std::string& name, LayerType t, bool cap = false) {
        ModelNode n; n.name = name; n.type = t; n.is_capture_target = cap; return n;
    };
    root.children.push_back(leaf("token_embd", LayerType::Embedding));
    ModelNode grp; grp.name = "layers"; grp.expanded = true;
    for (int i = 0; i < 4; ++i) {
        ModelNode blk; blk.name = "blk." + std::to_string(i);
        blk.children.push_back(leaf(blk.name + ".attn_norm", LayerType::RMSNorm));
        blk.children.push_back(leaf(blk.name + ".attn",      LayerType::Attention, i == 1));
        blk.children.push_back(leaf(blk.name + ".ffn_norm",  LayerType::RMSNorm));
        blk.children.push_back(leaf(blk.name + ".ffn_out",   LayerType::MLP));
        grp.children.push_back(blk);
    }
    root.children.push_back(grp);
    root.children.push_back(leaf("output_norm", LayerType::RMSNorm));
    root.children.push_back(leaf("output",      LayerType::Output));
    hook.set_topology(std::move(root));

    const struct { const char* name; LayerType type; double lat; float mx; float sp; } sim[] = {
        { "token_embd",      LayerType::Embedding, 0.21, 1.3f, 0.12f },
        { "blk.0.attn_norm", LayerType::RMSNorm,   0.02, 0.9f, 0.00f },
        { "blk.0.attn",      LayerType::Attention,  1.14, 3.2f, 0.54f },
        { "blk.0.ffn_norm",  LayerType::RMSNorm,   0.02, 0.8f, 0.00f },
        { "blk.0.ffn_out",   LayerType::MLP,        0.87, 5.1f, 0.61f },
        { "blk.1.attn_norm", LayerType::RMSNorm,   0.02, 0.9f, 0.00f },
        { "blk.1.attn",      LayerType::Attention,  1.19, 6.5f, 0.49f },
        { "blk.1.ffn_norm",  LayerType::RMSNorm,   0.02, 0.7f, 0.00f },
        { "blk.1.ffn_out",   LayerType::MLP,        0.91, 4.8f, 0.58f },
        { "blk.2.attn",      LayerType::Attention,  1.22, 2.9f, 0.51f },
        { "blk.2.ffn_out",   LayerType::MLP,        0.89, 4.3f, 0.60f },
        { "output_norm",     LayerType::RMSNorm,   0.03, 1.1f, 0.00f },
    };
    for (size_t i = 0; i < std::size(sim); ++i) {
        TelemetryPacket pkt;
        pkt.id         = 100 + i;
        pkt.timestamp  = std::chrono::system_clock::now();
        pkt.layer_name = sim[i].name;
        pkt.layer_type = sim[i].type;
        pkt.device     = ComputeDevice::CPU;
        pkt.latency_ms = sim[i].lat;
        pkt.tensor_stats.shape   = {1, 32, 4096};
        pkt.tensor_stats.dtype   = "float16";
        pkt.tensor_stats.mean    = 0.031f;
        pkt.tensor_stats.max_val = sim[i].mx;
        pkt.tensor_stats.min_val = -sim[i].mx * 0.9f;
        pkt.tensor_stats.sparsity = sim[i].sp;
        hook.packets().push(pkt);
        if (sim[i].mx > 6.0f)
            hook.anomalies().push({pkt.timestamp,
                "Outlier Feature " + pkt.layer_name + ": Max = " +
                std::to_string(sim[i].mx) + " > 6.0", false});
    }
}

// ── Real model inference ──────────────────────────────────────────────────────

static void run_model(ModelHook& hook, const char* model_path, const char* prompt) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "[prism] ERROR: cannot load " << model_path << "\n";
        llama_backend_free();
        return;
    }

    hook.set_topology(ModelHook::build_topology(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 512;
    cparams.cb_eval           = ModelHook::ggml_eval_cb;
    cparams.cb_eval_user_data = &hook;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n = -llama_tokenize(vocab, prompt, (int)std::strlen(prompt),
                            nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt, (int)std::strlen(prompt),
                   tokens.data(), n, true, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    if (llama_decode(ctx, batch) != 0)
        std::cerr << "[prism] WARNING: llama_decode returned error\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // ── Argument parsing ──────────────────────────────────────────────────────
    const char* model_path  = nullptr;
    const char* prompt      = "The quick brown fox jumps over the lazy dog";
    const char* save_path   = nullptr;
    const char* replay_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        if (!std::strcmp(argv[i], "--replay") && i + 1 < argc) {
            replay_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--save") && i + 1 < argc) {
            save_path = argv[++i];
        } else if (!model_path && argv[i][0] != '-') {
            model_path = argv[i];
        } else if (model_path && argv[i][0] != '-') {
            prompt = argv[i];
        }
    }

    ModelHook hook;
    std::thread inf_thread;

    if (replay_path) {
        // ── Replay mode ───────────────────────────────────────────────────────
        std::cout << "Loading trace: " << replay_path << " ...\n";
        if (!prism_trace::load(hook, replay_path)) {
            std::cerr << "ERROR: cannot load trace file: " << replay_path << "\n";
            return 1;
        }
        std::cout << "Loaded " << hook.packets().size() << " packets, "
                  << hook.anomalies().size() << " anomalies.\n";

    } else if (model_path) {
        // ── Live inference mode ───────────────────────────────────────────────
        inf_thread = std::thread([&]() {
            run_model(hook, model_path, prompt);
            // Save trace if requested
            if (save_path) {
                if (prism_trace::save(hook, save_path))
                    std::cerr << "[prism] Trace saved to " << save_path << "\n";
                else
                    std::cerr << "[prism] ERROR: could not save trace\n";
            }
        });

    } else {
        // ── Demo mode ─────────────────────────────────────────────────────────
        seed_synthetic(hook);
    }

    // Brief pause so the inference thread can set topology before TUI builds tree
    std::this_thread::sleep_for(std::chrono::milliseconds(80));

    TuiApp app(hook);
    app.run();

    if (inf_thread.joinable()) inf_thread.join();
    return 0;
}
