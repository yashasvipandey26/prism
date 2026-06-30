#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// ── Layer classification ─────────────────────────────────────────────────────

enum class LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    RMSNorm,
    Output,
    Unknown
};

enum class ComputeDevice {
    CPU,
    CUDA,
    Metal,
    Unknown
};

inline const char* layer_type_str(LayerType t) {
    switch (t) {
        case LayerType::Embedding:  return "Embedding";
        case LayerType::Attention:  return "Attn (Self)";
        case LayerType::MLP:        return "MLP (SwiGLU)";
        case LayerType::LayerNorm:  return "LayerNorm";
        case LayerType::RMSNorm:    return "RMSNorm";
        case LayerType::Output:     return "Output";
        default:                    return "Unknown";
    }
}

inline const char* device_str(ComputeDevice d) {
    switch (d) {
        case ComputeDevice::CPU:    return "CPU";
        case ComputeDevice::CUDA:   return "CUDA";
        case ComputeDevice::Metal:  return "Metal";
        default:                    return "Unknown";
    }
}

// ── Tensor statistics captured per layer ────────────────────────────────────

struct TensorStats {
    std::vector<int64_t> shape;
    std::string dtype;       // "float32", "float16", "int8", …
    float mean     = 0.0f;
    float max_val  = 0.0f;
    float min_val  = 0.0f;
    float sparsity = 0.0f;  // fraction of |x| < 1e-6

    std::string shape_str() const {
        if (shape.empty()) return "[]";
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        return s + "]";
    }
};

// ── Attention matrix snapshot ────────────────────────────────────────────────

struct AttentionMatrix {
    int num_heads = 0;
    int seq_len   = 0;
    std::vector<float> weights;  // flattened [num_heads][seq_len][seq_len]

    float get(int head, int row, int col) const {
        if (weights.empty() || head >= num_heads || row >= seq_len || col >= seq_len)
            return 0.0f;
        return weights[head * seq_len * seq_len + row * seq_len + col];
    }
};

// ── Primary telemetry record stored in the ring buffer ─────────────────────

struct TelemetryPacket {
    uint64_t id = 0;
    std::chrono::system_clock::time_point timestamp;

    std::string    layer_name;
    LayerType      layer_type = LayerType::Unknown;
    ComputeDevice  device     = ComputeDevice::Unknown;
    double         latency_ms = 0.0;

    TensorStats    tensor_stats;

    bool            has_attention = false;
    AttentionMatrix attention;
};

// ── Anomaly record ───────────────────────────────────────────────────────────

struct AnomalyRecord {
    std::chrono::system_clock::time_point timestamp;
    std::string message;
    bool is_error = false;  // false → warning, true → error
};

// ── Model topology node (tree) ───────────────────────────────────────────────

struct ModelNode {
    std::string name;
    LayerType   type     = LayerType::Unknown;
    bool        expanded = false;
    bool        is_capture_target = false;

    std::vector<ModelNode> children;

    bool is_leaf() const { return children.empty(); }
};
