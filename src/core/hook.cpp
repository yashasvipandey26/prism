#include "core/hook.h"

#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────

static bool name_has(const char* name, const char* needle) {
    return name && needle && std::strstr(name, needle) != nullptr;
}

// ── Static eval callback ──────────────────────────────────────────────────────
//
// Called by ggml for every tensor in the compute graph:
//   ask=true  → before computation; return true to request the post-computation call
//   ask=false → after computation;  tensor->data is valid; return false to continue

bool ModelHook::ggml_eval_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* hook = static_cast<ModelHook*>(user_data);
    if (ask) {
        bool capture = hook->should_capture(t->name);
        if (capture) hook->on_tensor_before(t->name);
        return capture;
    }
    hook->on_tensor_after(t);
    return false;
}

// ── Capture filter ────────────────────────────────────────────────────────────

bool ModelHook::should_capture(const char* name) const {
    if (!name || name[0] == '\0') return false;

    // Intercept key layer-output tensors; skip raw weight/bias ops.
    static const char* const patterns[] = {
        "attn_output",  // attention projection output
        "ffn_out",      // MLP output
        "attn_norm",    // pre-attention normalisation
        "ffn_norm",     // pre-FFN normalisation
        "token_embd",   // embedding lookup
        "inp_embd",     // input embedding
        "output_norm",  // final layer norm
        "blk.",         // catch-all for block intermediate tensors
        nullptr
    };
    for (const char* const* p = patterns; *p; ++p)
        if (name_has(name, *p)) return true;
    return false;
}

// ── Before tensor (ask=true path) ─────────────────────────────────────────────

void ModelHook::on_tensor_before(const char* name) {
    std::lock_guard<std::mutex> lk(timer_mu_);
    timers_[name] = std::chrono::high_resolution_clock::now();
}

// ── After tensor (ask=false path) ─────────────────────────────────────────────

void ModelHook::on_tensor_after(struct ggml_tensor* t) {
    auto now = std::chrono::high_resolution_clock::now();

    double latency_ms = 0.0;
    {
        std::lock_guard<std::mutex> lk(timer_mu_);
        auto it = timers_.find(t->name);
        if (it != timers_.end()) {
            latency_ms = std::chrono::duration<double, std::milli>(now - it->second).count();
            timers_.erase(it);
        }
    }

    TelemetryPacket pkt;
    pkt.id           = next_id_.fetch_add(1);
    pkt.timestamp    = std::chrono::system_clock::now();
    pkt.layer_name   = t->name;
    pkt.layer_type   = classify(t->name);
    pkt.device       = device_of(t);
    pkt.latency_ms   = latency_ms;
    pkt.tensor_stats = compute_stats(t);

    maybe_flag_anomaly(pkt);
    packets_.push(pkt);
}

// ── Layer-type classification ─────────────────────────────────────────────────

LayerType ModelHook::classify(const char* name) {
    if (!name) return LayerType::Unknown;
    if (name_has(name, "token_embd") || name_has(name, "inp_embd"))
        return LayerType::Embedding;
    if (name_has(name, "attn_norm") || name_has(name, "ffn_norm") || name_has(name, "output_norm"))
        return LayerType::RMSNorm;
    if (name_has(name, "attn"))
        return LayerType::Attention;
    if (name_has(name, "ffn") || name_has(name, "mlp"))
        return LayerType::MLP;
    if (name_has(name, "output"))
        return LayerType::Output;
    return LayerType::Unknown;
}

// ── Device detection ──────────────────────────────────────────────────────────

ComputeDevice ModelHook::device_of(struct ggml_tensor* t) {
    if (!t || !t->buffer) return ComputeDevice::CPU;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    if (!buft) return ComputeDevice::CPU;
    const char* bname = ggml_backend_buft_name(buft);
    if (!bname) return ComputeDevice::CPU;
    if (std::strstr(bname, "CUDA"))  return ComputeDevice::CUDA;
    if (std::strstr(bname, "Metal")) return ComputeDevice::Metal;
    return ComputeDevice::CPU;
}

// ── Activation statistics (CPU tensors only) ──────────────────────────────────

TensorStats ModelHook::compute_stats(struct ggml_tensor* t) {
    TensorStats s;

    // Shape: llama.cpp stores dims as ne[0]=fastest, so reverse for display
    int ndim = ggml_n_dims(t);
    s.shape.reserve(ndim);
    for (int i = ndim - 1; i >= 0; --i)
        s.shape.push_back(t->ne[i]);

    s.dtype = ggml_type_name(t->type);

    // Skip stat computation for GPU-resident tensors (would need expensive sync)
    if (!t->data) return s;
    if (t->buffer) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        if (buft) {
            const char* bname = ggml_backend_buft_name(buft);
            if (bname && (std::strstr(bname, "CUDA") || std::strstr(bname, "Metal")))
                return s;
        }
    }

    size_t n = ggml_nelements(t);
    if (n == 0) return s;

    float mn = 1e30f, mx = -1e30f, sum = 0.0f;
    size_t zeros = 0;

    auto accumulate = [&](float v) {
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (std::fabs(v) < kSparsityEps) ++zeros;
    };

    if (t->type == GGML_TYPE_F32) {
        const auto* data = static_cast<const float*>(t->data);
        for (size_t i = 0; i < n; ++i) accumulate(data[i]);
    } else if (t->type == GGML_TYPE_F16) {
        const auto* data = static_cast<const ggml_fp16_t*>(t->data);
        for (size_t i = 0; i < n; ++i) accumulate(ggml_fp16_to_fp32(data[i]));
    } else {
        return s;  // quantised types: record shape/dtype only
    }

    s.mean     = sum / static_cast<float>(n);
    s.min_val  = mn;
    s.max_val  = mx;
    s.sparsity = static_cast<float>(zeros) / static_cast<float>(n);
    return s;
}

// ── Anomaly detection ─────────────────────────────────────────────────────────

void ModelHook::maybe_flag_anomaly(const TelemetryPacket& pkt) {
    const auto& s = pkt.tensor_stats;

    // Outlier activations (numerical instability risk)
    if (s.max_val > kAnomalyMaxThresh && s.max_val < 1e29f) {
        anomalies_.push({
            pkt.timestamp,
            "Outlier Feature " + pkt.layer_name + ": Max = " +
                std::to_string(s.max_val) + " > " + std::to_string(kAnomalyMaxThresh),
            false  // warning
        });
    }

    // Unexpected CPU fallback for attention/MLP layers
    if (pkt.device == ComputeDevice::CPU &&
        (pkt.layer_type == LayerType::Attention || pkt.layer_type == LayerType::MLP)) {
        anomalies_.push({
            pkt.timestamp,
            "CPU Fallback: " + pkt.layer_name + " on CPU (expected accelerator)",
            true  // error
        });
    }
}

// ── Topology builder ──────────────────────────────────────────────────────────

ModelNode ModelHook::build_topology(const struct llama_model* model) {
    ModelNode root;

    char desc[256] = "model";
    llama_model_desc(model, desc, sizeof(desc));
    root.name     = desc;
    root.expanded = true;

    const int n_layers = llama_model_n_layer(model);

    // Embedding
    {
        ModelNode embd;
        embd.name = "token_embd";
        embd.type = LayerType::Embedding;
        root.children.push_back(embd);
    }

    // Transformer blocks
    {
        ModelNode grp;
        grp.name     = "layers";
        grp.expanded = true;

        for (int i = 0; i < n_layers; ++i) {
            const std::string prefix = "blk." + std::to_string(i);
            ModelNode blk;
            blk.name = prefix;

            auto make_child = [&](const std::string& suffix, LayerType type) {
                ModelNode n;
                n.name = prefix + "." + suffix;
                n.type = type;
                return n;
            };

            blk.children.push_back(make_child("attn_norm",  LayerType::RMSNorm));
            blk.children.push_back(make_child("attn",       LayerType::Attention));
            blk.children.push_back(make_child("ffn_norm",   LayerType::RMSNorm));
            blk.children.push_back(make_child("ffn_out",    LayerType::MLP));

            grp.children.push_back(blk);
        }
        root.children.push_back(grp);
    }

    // Output head
    {
        ModelNode on; on.name = "output_norm"; on.type = LayerType::RMSNorm;
        ModelNode out; out.name = "output";    out.type = LayerType::Output;
        root.children.push_back(on);
        root.children.push_back(out);
    }

    return root;
}
