#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <optional>
// ─── Enums ───────────────────────────────────────────────────────────────────

enum class LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    Output,
    Unknown
};

enum class DeviceType {
    CPU,
    CUDA,
    Metal
};

// ─── Tensor Metadata ─────────────────────────────────────────────────────────

struct TensorShape {
    std::vector<int64_t> dims;

    std::string to_string() const {
        std::string s = "[";
        for (size_t i = 0; i < dims.size(); i++) {
            if (i) s += ", ";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

struct TensorStats {
    float mean    = 0.f;
    float std_dev = 0.f;
    float min_val = 0.f;
    float max_val = 0.f;
    float sparsity = 0.f;   // fraction of near-zero elements
    std::string dtype = "float32";
};

// ─── Attention Data ───────────────────────────────────────────────────────────

struct AttentionSnapshot {
    int num_heads   = 0;
    int seq_len     = 0;
    // attn_weights[head][row][col]  (kept small: max 32 heads × 64 tokens)
    std::vector<std::vector<std::vector<float>>> attn_weights;
    std::vector<std::string> tokens;
};

// ─── Layer Record ─────────────────────────────────────────────────────────────

struct LayerRecord {
    uint64_t    id          = 0;
    std::string name;
    LayerType   type        = LayerType::Unknown;
    DeviceType  device      = DeviceType::CPU;
    int         device_idx  = 0;
    bool        is_fallback = false;   // e.g. CPU fallback from CUDA OOM

    TensorShape input_shape;
    TensorShape output_shape;
    TensorStats stats;

    double latency_ms = 0.0;          // wall-clock time for this layer
    double mean_latency_ms = 0.0;     // rolling average

    std::chrono::system_clock::time_point timestamp;
    std::optional<AttentionSnapshot> attn;

    // Anomaly flags
    bool has_outlier   = false;
    bool has_nan_inf   = false;
    std::string anomaly_msg;
};

// ─── Model Topology Node ─────────────────────────────────────────────────────

struct TopoNode {
    std::string name;
    LayerType   type     = LayerType::Unknown;
    bool        expanded = false;
    bool        selected = false;
    int         depth    = 0;
    std::vector<TopoNode> children;
};
