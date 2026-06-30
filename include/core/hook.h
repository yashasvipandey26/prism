#pragma once

#include "core/ring_buffer.h"
#include "core/telemetry.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

// Forward-declare ggml/llama types so callers don't need to pull in those headers.
struct ggml_tensor;
struct llama_model;

static constexpr size_t PACKET_BUF_CAP  = 256;
static constexpr size_t ANOMALY_BUF_CAP = 64;

// ── ModelHook ─────────────────────────────────────────────────────────────────
//
// Non-invasive instrumentation layer for llama.cpp inference.
//
// Usage:
//   ModelHook hook;
//
//   llama_context_params cparams = llama_context_default_params();
//   cparams.cb_eval           = ModelHook::ggml_eval_cb;
//   cparams.cb_eval_user_data = &hook;
//
//   llama_context* ctx = llama_init_from_model(model, cparams);
//   hook.set_topology(ModelHook::build_topology(model));
//
//   // After llama_decode(...), read hook.packets() / hook.anomalies().
//
class ModelHook {
public:
    ModelHook() = default;

    // ── ggml eval callback ────────────────────────────────────────────────────
    // Set in llama_context_params::cb_eval.
    // ask=true  → returning true means "call me again after this tensor is computed"
    // ask=false → tensor data is ready; we read and record stats, then return false
    static bool ggml_eval_cb(struct ggml_tensor* t, bool ask, void* user_data);

    // ── Topology ──────────────────────────────────────────────────────────────
    // Reconstruct the model's layer tree from llama metadata.
    static ModelNode build_topology(const struct llama_model* model);

    // ── Data access (read by TUI) ─────────────────────────────────────────────
    RingBuffer<TelemetryPacket, PACKET_BUF_CAP>&  packets()   { return packets_;   }
    RingBuffer<AnomalyRecord,   ANOMALY_BUF_CAP>& anomalies() { return anomalies_; }

    void             set_topology(ModelNode n) { topology_ = std::move(n); }
    const ModelNode& topology() const          { return topology_; }

    // ── Real attention access ─────────────────────────────────────────────────
    // Populated from the KQ_soft_max tensor captured during forward pass.
    bool            has_real_attention() const { return has_real_attn_.load(); }
    AttentionMatrix copy_attention() const {
        std::lock_guard<std::mutex> lk(attn_mu_);
        return real_attn_;
    }

private:
    // Internal dispatch
    bool should_capture(const char* name) const;
    void on_tensor_before(const char* name);
    void on_tensor_after(struct ggml_tensor* t);
    void capture_attention(struct ggml_tensor* t);

    // Helpers
    static LayerType    classify(const char* name);
    static ComputeDevice device_of(struct ggml_tensor* t);
    static TensorStats   compute_stats(struct ggml_tensor* t);
    void                 maybe_flag_anomaly(const TelemetryPacket& pkt);

    // ── State ─────────────────────────────────────────────────────────────────
    RingBuffer<TelemetryPacket, PACKET_BUF_CAP>  packets_;
    RingBuffer<AnomalyRecord,   ANOMALY_BUF_CAP> anomalies_;
    ModelNode topology_;

    std::atomic<uint64_t> next_id_{100};

    // Tensor-name → start-time, to derive latency between ask=true and ask=false
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers_;
    std::mutex timer_mu_;

    // Captured attention matrix (latest KQ softmax from any layer)
    AttentionMatrix    real_attn_;
    std::atomic<bool>  has_real_attn_{false};
    mutable std::mutex attn_mu_;

    static constexpr float kSparsityEps       = 1e-6f;
    static constexpr float kAnomalyMaxThresh  = 6.0f;
};
