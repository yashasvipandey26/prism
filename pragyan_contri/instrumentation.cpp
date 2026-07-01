#include "instrumentation.h"
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <sstream>
#include <algorithm>

// ─── Singleton ────────────────────────────────────────────────────────────────

InstrumentationEngine& InstrumentationEngine::instance() {
    static InstrumentationEngine eng;
    return eng;
}

// ─── Listener management ──────────────────────────────────────────────────────

void InstrumentationEngine::add_listener(LayerCallback cb) {
    std::lock_guard<std::mutex> lk(listener_mtx_);
    listeners_.push_back(std::move(cb));
}

void InstrumentationEngine::record_layer(LayerRecord rec) {
    buffer_.push(rec);
    std::lock_guard<std::mutex> lk(listener_mtx_);
    for (auto& cb : listeners_) cb(rec);
}

// ─── Latency EMA (α = 0.1) ───────────────────────────────────────────────────

double InstrumentationEngine::update_latency(const std::string& name, double ms) {
    std::lock_guard<std::mutex> lk(latency_mtx_);
    auto it = latency_ema_.find(name);
    if (it == latency_ema_.end()) {
        latency_ema_[name] = ms;
        return ms;
    }
    constexpr double alpha = 0.1;
    it->second = alpha * ms + (1.0 - alpha) * it->second;
    return it->second;
}

// ─── Synthetic simulation ─────────────────────────────────────────────────────

static LayerType random_layer_type(std::mt19937& rng) {
    static const std::vector<LayerType> types = {
        LayerType::Attention, LayerType::MLP,
        LayerType::LayerNorm, LayerType::Embedding
    };
    return types[rng() % types.size()];
}

static std::string layer_type_name(LayerType t) {
    switch (t) {
        case LayerType::Attention:  return "attn";
        case LayerType::MLP:        return "mlp";
        case LayerType::LayerNorm:  return "norm";
        case LayerType::Embedding:  return "embed_tokens";
        default:                    return "unknown";
    }
}

// Build a fake attention matrix (causal-ish pattern).
static AttentionSnapshot make_attn(const std::vector<std::string>& tokens,
                                   int num_heads, std::mt19937& rng)
{
    AttentionSnapshot snap;
    snap.tokens    = tokens;
    snap.num_heads = num_heads;
    snap.seq_len   = static_cast<int>(tokens.size());

    std::uniform_real_distribution<float> dist(0.f, 1.f);

    snap.attn_weights.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        snap.attn_weights[h].resize(snap.seq_len, std::vector<float>(snap.seq_len, 0.f));
        for (int r = 0; r < snap.seq_len; ++r) {
            float sum = 0.f;
            for (int c = 0; c <= r; ++c) {
                float w = dist(rng);
                // Boost diagonal to simulate self-attention peak.
                if (c == r) w *= 2.5f;
                snap.attn_weights[h][r][c] = w;
                sum += w;
            }
            // Softmax-normalise.
            if (sum > 0.f)
                for (int c = 0; c <= r; ++c)
                    snap.attn_weights[h][r][c] /= sum;
        }
    }
    return snap;
}

LayerRecord InstrumentationEngine::make_synthetic_record(
        uint64_t id, const std::string& name, LayerType type,
        DeviceType dev, const std::vector<std::string>& tokens)
{
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> latency_dist(0.3f, 6.f);
    std::uniform_real_distribution<float> stat_dist(0.f, 1.f);
    std::uniform_int_distribution<int>    dim_dist(64, 4096);

    LayerRecord r;
    r.id        = id;
    r.name      = name;
    r.type      = type;
    r.device    = dev;
    r.device_idx = 0;
    r.timestamp = std::chrono::system_clock::now();

    // Simulate CPU fallback occasionally.
    r.is_fallback = (dev == DeviceType::CPU && type == LayerType::LayerNorm && id % 7 == 0);

    // Shapes.
    int seq  = static_cast<int>(tokens.size());
    int hidden = (type == LayerType::Embedding) ? 4096 : dim_dist(rng) / 64 * 64;
    r.input_shape  = {{1, static_cast<int64_t>(seq), hidden}};
    r.output_shape = r.input_shape;

    // Stats.
    r.stats.mean     = stat_dist(rng) * 0.2f - 0.1f;
    r.stats.std_dev  = stat_dist(rng) * 0.5f;
    r.stats.min_val  = r.stats.mean - r.stats.std_dev * 3.f;
    r.stats.max_val  = r.stats.mean + r.stats.std_dev * 3.f;
    r.stats.sparsity = stat_dist(rng);
    r.stats.dtype    = "float16";

    // Latency.
    double raw_lat   = latency_dist(rng);
    r.latency_ms     = raw_lat;
    r.mean_latency_ms = update_latency(name, raw_lat);

    // Attention snapshot for attention layers.
    if (type == LayerType::Attention && seq > 0) {
        r.attn = make_attn(tokens, 8, rng);
    }

    // Anomaly detection.
    if (r.stats.max_val > 6.f) {
        r.has_outlier  = true;
        r.anomaly_msg  = "Outlier Feature " + name + ": Max > 6.0";
    }
    if (r.is_fallback) {
        if (!r.anomaly_msg.empty()) r.anomaly_msg += " | ";
        r.anomaly_msg += std::string("CUDA OOM Fallback: Processing ") + name + " on CPU Host Memory.";
    }

    return r;
}

void InstrumentationEngine::simulation_loop(
        std::string model_name, std::vector<std::string> tokens)
{
    model_name_ = model_name;

    // Build topology.
    root_topo_.name     = model_name;
    root_topo_.type     = LayerType::Unknown;
    root_topo_.expanded = true;
    root_topo_.depth    = 0;
    root_topo_.children.clear();

    // embed_tokens.
    TopoNode embed;
    embed.name  = "embed_tokens";
    embed.type  = LayerType::Embedding;
    embed.depth = 1;
    root_topo_.children.push_back(embed);

    // transformer layers (32 blocks).
    TopoNode layers_node;
    layers_node.name     = "layers";
    layers_node.type     = LayerType::Unknown;
    layers_node.expanded = true;
    layers_node.depth    = 1;

    for (int i = 0; i < 32; ++i) {
        TopoNode block;
        block.name     = "layers." + std::to_string(i);
        block.expanded = (i < 2);
        block.depth    = 2;

        TopoNode attn_node;
        attn_node.name  = "layers." + std::to_string(i) + ".attn";
        attn_node.type  = LayerType::Attention;
        attn_node.depth = 3;

        TopoNode mlp_node;
        mlp_node.name  = "layers." + std::to_string(i) + ".mlp";
        mlp_node.type  = LayerType::MLP;
        mlp_node.depth = 3;

        block.children.push_back(attn_node);
        block.children.push_back(mlp_node);
        layers_node.children.push_back(block);
    }
    root_topo_.children.push_back(layers_node);

    // norm + lm_head.
    TopoNode norm;
    norm.name  = "norm";
    norm.type  = LayerType::LayerNorm;
    norm.depth = 1;
    root_topo_.children.push_back(norm);

    TopoNode lm_head;
    lm_head.name  = "lm_head";
    lm_head.type  = LayerType::Output;
    lm_head.depth = 1;
    root_topo_.children.push_back(lm_head);

    // ── Emit records in a loop simulating forward pass ──────────────────
    uint64_t id = 1;

    // Helper layer sequence repeated per token generation step.
    struct LayerDef { std::string name; LayerType type; DeviceType dev; };
    std::vector<LayerDef> seq;

    seq.push_back({"embed_tokens", LayerType::Embedding, DeviceType::CUDA});
    for (int i = 0; i < 32; ++i) {
        seq.push_back({"layers." + std::to_string(i) + ".self_attn",
                        LayerType::Attention, DeviceType::CUDA});
        seq.push_back({"layers." + std::to_string(i) + ".mlp",
                        LayerType::MLP, DeviceType::CUDA});
        seq.push_back({"layers." + std::to_string(i) + ".input_layernorm",
                        LayerType::LayerNorm,
                        (i % 7 == 0) ? DeviceType::CPU : DeviceType::CUDA});
    }
    seq.push_back({"norm",    LayerType::LayerNorm, DeviceType::CUDA});
    seq.push_back({"lm_head", LayerType::Output,    DeviceType::CUDA});

    std::size_t pos = 0;
    while (simulating_.load()) {
        auto& def = seq[pos % seq.size()];
        auto rec  = make_synthetic_record(id++, def.name, def.type, def.dev, tokens);
        record_layer(std::move(rec));

        // Vary speed: attention/MLP slower, norms faster.
        int delay_ms = (def.type == LayerType::Attention || def.type == LayerType::MLP)
                       ? 120 : 40;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        ++pos;
    }
}

void InstrumentationEngine::start_simulation(
        const std::string& model_name, const std::vector<std::string>& tokens)
{
    if (simulating_.load()) return;
    simulating_.store(true);
    sim_thread_ = std::thread([this, model_name, tokens]() {
        simulation_loop(model_name, tokens);
    });
}

void InstrumentationEngine::stop_simulation() {
    simulating_.store(false);
    if (sim_thread_.joinable()) sim_thread_.join();
}
