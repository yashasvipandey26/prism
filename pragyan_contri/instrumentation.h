#pragma once
#include <sys/types.h>
#include "types.h"
#include "ring_buffer.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>

// ─── Hook callback types ──────────────────────────────────────────────────────

using LayerCallback = std::function<void(const LayerRecord&)>;

// ─── Instrumentation Engine ───────────────────────────────────────────────────
//
//  Non-invasive interception via:
//    1.  Pre/Post hooks registered per layer name (regex match supported)
//    2.  A synthetic "simulated model" when no real model is attached —
//        used for demo / testing without an actual inference engine.
//
//  Real usage:
//    - For llama.cpp: link against it, subclass or patch ggml_compute_forward
//      to call InstrumentationEngine::record_layer() before/after each op.
//    - For ONNX Runtime: use its custom logger / EP hooks.
//    - For PyTorch via libtorch: register forward hooks on each module.
//
//  All captured data goes into the global RingBuffer<LayerRecord, 256>.

constexpr std::size_t RING_CAPACITY = 256;

class InstrumentationEngine {
public:
    static InstrumentationEngine& instance();

    // Register a callback fired for every captured LayerRecord.
    void add_listener(LayerCallback cb);

    // Called by hooks (real or synthetic) to record a layer event.
    void record_layer(LayerRecord rec);

    // Access the ring buffer (read-only snapshot).
    RingBuffer<LayerRecord, RING_CAPACITY>& buffer() { return buffer_; }

    // Running counter.
    uint64_t next_id() { return id_counter_++; }

    // Latency tracker: compute rolling mean per layer name.
    double update_latency(const std::string& name, double ms);

    // Start/stop synthetic simulation (for demo without a real model).
    void start_simulation(const std::string& model_name, const std::vector<std::string>& tokens);
    void stop_simulation();
    bool is_simulating() const { return simulating_.load(); }

    // Model topology (populated by simulation or real hook scan).
    TopoNode& topology() { return root_topo_; }

    std::string model_name() const { return model_name_; }

private:
    InstrumentationEngine() = default;

    RingBuffer<LayerRecord, RING_CAPACITY> buffer_;
    std::vector<LayerCallback> listeners_;
    std::mutex listener_mtx_;
    std::atomic<uint64_t> id_counter_{1};

    // Rolling latency per layer name.
    std::unordered_map<std::string, double> latency_ema_;
    std::mutex latency_mtx_;

    // Simulation thread.
    std::atomic<bool> simulating_{false};
    std::thread sim_thread_;

    TopoNode root_topo_;
    std::string model_name_ = "No Model Attached";

    void simulation_loop(std::string model_name, std::vector<std::string> tokens);
    LayerRecord make_synthetic_record(uint64_t id, const std::string& name,
                                      LayerType type, DeviceType dev,
                                      const std::vector<std::string>& tokens);
};