#include "core/hook.h"
#include "core/ring_buffer.h"
#include "core/telemetry.h"

#include <llama.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ── Formatting helpers ────────────────────────────────────────────────────────

static std::string fmt_time(std::chrono::system_clock::time_point tp) {
    auto t  = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  tp.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return std::string(buf) + "." + std::to_string(ms.count() / 10);
}

static void print_divider(char c = '-', int w = 72) {
    std::cout << std::string(w, c) << "\n";
}

// ── Synthetic smoke-test (no model file required) ─────────────────────────────

static void run_synthetic_test() {
    std::cout << "\n[Synthetic mode — no model path given]\n\n";

    ModelHook hook;

    // Build a fake topology
    ModelNode root;
    root.name = "llama-3-8b (synthetic)";
    root.expanded = true;

    for (const char* name : {"token_embd", "blk.0.attn", "blk.0.ffn_out",
                              "blk.1.attn", "blk.1.ffn_out", "output_norm", "output"}) {
        ModelNode n;
        n.name = name;
        if      (std::strstr(name, "embd"))   n.type = LayerType::Embedding;
        else if (std::strstr(name, "attn"))   n.type = LayerType::Attention;
        else if (std::strstr(name, "ffn"))    n.type = LayerType::MLP;
        else if (std::strstr(name, "norm"))   n.type = LayerType::RMSNorm;
        else if (std::strstr(name, "output")) n.type = LayerType::Output;
        root.children.push_back(n);
    }
    hook.set_topology(std::move(root));

    // Simulate what the eval callback would produce
    const struct {
        const char* name;
        LayerType   type;
        double      lat_ms;
        float       max_val;
    } sim[] = {
        { "token_embd",     LayerType::Embedding, 0.21, 1.3f },
        { "blk.0.attn_norm",LayerType::RMSNorm,   0.02, 0.9f },
        { "blk.0.attn",     LayerType::Attention, 1.14, 3.2f },
        { "blk.0.ffn_norm", LayerType::RMSNorm,   0.02, 0.8f },
        { "blk.0.ffn_out",  LayerType::MLP,       0.87, 5.1f },
        { "blk.1.attn_norm",LayerType::RMSNorm,   0.02, 0.9f },
        { "blk.1.attn",     LayerType::Attention, 1.19, 6.5f },  // triggers anomaly
        { "blk.1.ffn_norm", LayerType::RMSNorm,   0.02, 0.7f },
        { "blk.1.ffn_out",  LayerType::MLP,       0.91, 4.8f },
        { "output_norm",    LayerType::RMSNorm,   0.03, 1.1f },
    };

    for (size_t i = 0; i < std::size(sim); ++i) {
        TelemetryPacket pkt;
        pkt.id           = 100 + i;
        pkt.timestamp    = std::chrono::system_clock::now();
        pkt.layer_name   = sim[i].name;
        pkt.layer_type   = sim[i].type;
        pkt.device       = ComputeDevice::CPU;
        pkt.latency_ms   = sim[i].lat_ms;
        pkt.tensor_stats.shape   = {1, 32, 4096};
        pkt.tensor_stats.dtype   = "float16";
        pkt.tensor_stats.mean    = 0.03f;
        pkt.tensor_stats.max_val = sim[i].max_val;
        pkt.tensor_stats.sparsity = 0.54f;

        hook.packets().push(pkt);

        if (sim[i].max_val > 6.0f) {
            hook.anomalies().push({
                pkt.timestamp,
                "Outlier Feature " + pkt.layer_name + ": Max = " +
                    std::to_string(sim[i].max_val) + " > 6.0",
                false
            });
        }
    }

    // ── Print live stream ─────────────────────────────────────────────────────
    std::cout << std::left
              << std::setw(5)  << "ID"
              << std::setw(14) << "TIMESTAMP"
              << std::setw(18) << "LAYER TYPE"
              << std::setw(24) << "LAYER NAME"
              << "LATENCY\n";
    print_divider();

    hook.packets().for_each([](const TelemetryPacket& p) {
        std::cout << std::left
                  << std::setw(5)  << p.id
                  << std::setw(14) << fmt_time(p.timestamp)
                  << std::setw(18) << layer_type_str(p.layer_type)
                  << std::setw(24) << p.layer_name
                  << std::fixed << std::setprecision(3) << p.latency_ms << " ms\n";
    });

    // ── Print anomalies ───────────────────────────────────────────────────────
    std::cout << "\nANOMALY LEDGER (" << hook.anomalies().size() << " record(s)):\n";
    print_divider();
    hook.anomalies().for_each([](const AnomalyRecord& a) {
        std::cout << (a.is_error ? "[ERR] " : "[WRN] ")
                  << fmt_time(a.timestamp) << "  " << a.message << "\n";
    });

    std::cout << "\nTopology root: " << hook.topology().name
              << "  (" << hook.topology().children.size() << " children)\n";
}

// ── Real model inference ──────────────────────────────────────────────────────

static void run_with_model(const char* model_path, const char* prompt) {
    std::cout << "\n[Model: " << model_path << "]\n";
    std::cout << "[Prompt: \"" << prompt << "\"]\n\n";

    // ── Backend init ──────────────────────────────────────────────────────────
    llama_backend_init();

    // ── Load model ────────────────────────────────────────────────────────────
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // offload all layers to GPU if available

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "ERROR: failed to load model from " << model_path << "\n";
        llama_backend_free();
        return;
    }

    // ── Build topology before attaching hook ──────────────────────────────────
    ModelHook hook;
    hook.set_topology(ModelHook::build_topology(model));

    std::cout << "Model loaded: " << hook.topology().name << "\n";
    std::cout << "Layers      : " << llama_model_n_layer(model) << "\n";
    std::cout << "Embedding   : " << llama_model_n_embd(model) << "\n\n";

    // ── Create context with hook callback ─────────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 512;
    cparams.cb_eval           = ModelHook::ggml_eval_cb;
    cparams.cb_eval_user_data = &hook;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "ERROR: failed to create llama context\n";
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    // ── Tokenise ──────────────────────────────────────────────────────────────
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt, (int)std::strlen(prompt),
                                         nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt, (int)std::strlen(prompt),
                       tokens.data(), n_prompt, true, true) < 0) {
        std::cerr << "ERROR: tokenisation failed\n";
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    std::cout << "Prompt tokens: " << n_prompt << "\n";
    std::cout << "Running forward pass (hook active)...\n\n";

    // ── Forward pass (one decode call — we want activations, not output tokens) ─
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "ERROR: llama_decode failed\n";
    }

    // ── Print captured telemetry ──────────────────────────────────────────────
    std::cout << "Packets captured: " << hook.packets().size() << "\n\n";

    std::cout << std::left
              << std::setw(5)  << "ID"
              << std::setw(14) << "TIMESTAMP"
              << std::setw(18) << "LAYER TYPE"
              << std::setw(8)  << "DEV"
              << std::setw(28) << "LAYER NAME"
              << "LATENCY\n";
    print_divider();

    hook.packets().for_each([](const TelemetryPacket& p) {
        std::cout << std::left
                  << std::setw(5)  << p.id
                  << std::setw(14) << fmt_time(p.timestamp)
                  << std::setw(18) << layer_type_str(p.layer_type)
                  << std::setw(8)  << device_str(p.device)
                  << std::setw(28) << p.layer_name
                  << std::fixed << std::setprecision(3) << p.latency_ms << " ms\n";
    });

    // ── Anomalies ─────────────────────────────────────────────────────────────
    if (hook.anomalies().size() > 0) {
        std::cout << "\nANOMALY LEDGER (" << hook.anomalies().size() << " record(s)):\n";
        print_divider();
        hook.anomalies().for_each([](const AnomalyRecord& a) {
            std::cout << (a.is_error ? "[ERR] " : "[WRN] ")
                      << fmt_time(a.timestamp) << "  " << a.message << "\n";
        });
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::cout << "PRISM - LLM Telemetry Platform v0.2\n";
    print_divider('=');

    if (argc >= 2) {
        const char* prompt = (argc >= 3) ? argv[2] : "Hello, world!";
        run_with_model(argv[1], prompt);
    } else {
        run_synthetic_test();
    }

    return 0;
}
