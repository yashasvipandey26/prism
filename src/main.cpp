#include "core/ring_buffer.h"
#include "core/telemetry.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string fmt_time(std::chrono::system_clock::time_point tp) {
    auto t  = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  tp.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    std::string s = buf;
    s += '.' + std::to_string(ms.count() / 10);   // two-digit ms
    return s;
}

// ── Synthetic data for smoke-test ────────────────────────────────────────────

static TelemetryPacket make_packet(uint64_t id, const char* name, LayerType type, double lat_ms) {
    TelemetryPacket pkt;
    pkt.id         = id;
    pkt.timestamp  = std::chrono::system_clock::now();
    pkt.layer_name = name;
    pkt.layer_type = type;
    pkt.device     = ComputeDevice::CPU;
    pkt.latency_ms = lat_ms;

    pkt.tensor_stats.shape    = {1, 32, 4096};
    pkt.tensor_stats.dtype    = "float16";
    pkt.tensor_stats.mean     = 0.03f;
    pkt.tensor_stats.max_val  = 6.21f;
    pkt.tensor_stats.min_val  = -5.87f;
    pkt.tensor_stats.sparsity = 0.542f;

    return pkt;
}

int main() {
    std::cout << "PRISM - LLM Telemetry Platform v0.1\n";
    std::cout << std::string(40, '=') << "\n\n";

    // ── Ring buffer smoke-test ──────────────────────────────────────────────
    constexpr size_t BUF_CAP = 256;
    RingBuffer<TelemetryPacket, BUF_CAP> buffer;

    std::cout << "Ring buffer capacity : " << buffer.capacity() << "\n";

    // Push synthetic packets
    const struct { const char* name; LayerType type; double lat; } layers[] = {
        { "embed_tokens",   LayerType::Embedding, 0.21 },
        { "layers.0.attn",  LayerType::Attention, 1.14 },
        { "layers.0.mlp",   LayerType::MLP,       0.87 },
        { "layers.1.attn",  LayerType::Attention, 1.19 },
        { "layers.1.mlp",   LayerType::MLP,       0.91 },
        { "layers.1.norm",  LayerType::RMSNorm,   0.02 },
    };
    for (size_t i = 0; i < std::size(layers); ++i) {
        buffer.push(make_packet(100 + i, layers[i].name, layers[i].type, layers[i].lat));
    }

    std::cout << "Packets in buffer    : " << buffer.size() << "\n\n";

    // ── Print live stream table ─────────────────────────────────────────────
    std::cout << std::left
              << std::setw(5)  << "ID"
              << std::setw(14) << "TIMESTAMP"
              << std::setw(16) << "LAYER TYPE"
              << std::setw(20) << "LAYER NAME"
              << std::setw(10) << "LATENCY"
              << "\n" << std::string(65, '-') << "\n";

    buffer.for_each([](const TelemetryPacket& p) {
        std::cout << std::left
                  << std::setw(5)  << p.id
                  << std::setw(14) << fmt_time(p.timestamp)
                  << std::setw(16) << layer_type_str(p.layer_type)
                  << std::setw(20) << p.layer_name
                  << std::fixed << std::setprecision(3) << p.latency_ms << " ms\n";
    });

    // ── Anomaly ring buffer ─────────────────────────────────────────────────
    RingBuffer<AnomalyRecord, 64> anomalies;
    anomalies.push({
        std::chrono::system_clock::now(),
        "Outlier Feature Layer 1: Max > 6.0",
        false
    });
    std::cout << "\nAnomalies logged     : " << anomalies.size() << "\n";

    // ── Model topology smoke-test ───────────────────────────────────────────
    ModelNode root;
    root.name = "llama-3-8b";
    root.expanded = true;

    ModelNode embed; embed.name = "embed_tokens"; embed.type = LayerType::Embedding;
    root.children.push_back(embed);

    ModelNode layers_group; layers_group.name = "layers"; layers_group.expanded = true;
    for (int i = 0; i < 3; ++i) {
        ModelNode layer; layer.name = "layers." + std::to_string(i); layer.expanded = (i == 1);
        ModelNode attn; attn.name = layer.name + ".attn"; attn.type = LayerType::Attention;
        ModelNode mlp;  mlp.name  = layer.name + ".mlp";  mlp.type  = LayerType::MLP;
        if (i == 1) attn.is_capture_target = true;
        layer.children.push_back(attn);
        layer.children.push_back(mlp);
        layers_group.children.push_back(layer);
    }
    root.children.push_back(layers_group);

    std::cout << "\nModel topology root  : " << root.name << "\n";
    std::cout << "  Children           : " << root.children.size() << "\n";
    std::cout << "  Layer groups       : " << layers_group.children.size() << "\n";

    std::cout << "\nAll data structures OK.\n";
    return 0;
}
