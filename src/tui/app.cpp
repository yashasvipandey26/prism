#include "tui/app.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <thread>

using namespace ftxui;

// ── Small utilities ───────────────────────────────────────────────────────────

static std::string fmt_time(std::chrono::system_clock::time_point tp) {
    auto tt = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  tp.time_since_epoch()) % 1000;
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&tt));
    std::string s(buf);
    s += '.';
    int ms2 = (int)(ms.count() / 10);
    s += (char)('0' + ms2 / 10);
    s += (char)('0' + ms2 % 10);
    return s;
}

static std::string trunc(const std::string& s, int w) {
    return s.size() <= (size_t)w ? s : s.substr(0, w - 1) + "…";
}

// Attention heatmap: map [0,1] → block character + colour
static const char* attn_glyph(float v) {
    if (v > 0.75f) return "██";
    if (v > 0.50f) return "▓▓";
    if (v > 0.25f) return "▒▒";
    if (v > 0.08f) return "░░";
    return "  ";
}

static Color attn_color(float v) {
    // Cool blue → warm yellow/red
    uint8_t r = (uint8_t)(std::min(1.0f, v * 2.0f) * 255);
    uint8_t g = (uint8_t)(std::max(0.0f, 1.0f - std::abs(v - 0.5f) * 2.5f) * 200);
    uint8_t b = (uint8_t)(std::max(0.0f, 1.0f - v * 2.5f) * 255);
    return Color::RGB(r, g, b);
}

// ── Constructor ───────────────────────────────────────────────────────────────

TuiApp::TuiApp(ModelHook& hook) : hook_(hook) {
    rebuild_tree();

    // Synthetic causal attention (used when no real attention is captured)
    const int seq = 8, nh = 4;
    synth_attn_.num_heads = nh;
    synth_attn_.seq_len   = seq;
    synth_attn_.weights.assign(nh * seq * seq, 0.0f);
    for (int h = 0; h < nh; ++h) {
        for (int r = 0; r < seq; ++r) {
            float sum = 0.0f;
            for (int c = 0; c <= r; ++c) {
                float w = std::exp(-(r - c) * (0.3f + h * 0.12f));
                synth_attn_.weights[h * seq * seq + r * seq + c] = w;
                sum += w;
            }
            for (int c = 0; c <= r; ++c)
                synth_attn_.weights[h * seq * seq + r * seq + c] /= sum;
        }
    }
    synth_tokens_ = {"[I]", "[want]", "[it]", "[to]", "[be]", "[kbrd]", "[driv]", "[EOS]"};

    // Collapse blk.N nodes by default so the tree isn't overwhelming
    for (const auto& fn : flat_) {
        if (fn.has_children && fn.depth >= 2)
            collapsed_.insert(fn.full_name);
    }
    rebuild_tree();
}

// ── Tree flattening ───────────────────────────────────────────────────────────

void TuiApp::flatten_node(const ModelNode& node, int depth) {
    FlatNode fn;
    fn.full_name         = node.name;
    fn.type              = node.type;
    fn.depth             = depth;
    fn.has_children      = !node.children.empty();
    fn.is_capture_target = node.is_capture_target;
    flat_.push_back(fn);
    // Don't recurse into collapsed nodes
    if (collapsed_.count(node.name)) return;
    for (const auto& child : node.children)
        flatten_node(child, depth + 1);
}

void TuiApp::rebuild_tree() {
    flat_.clear();
    const ModelNode& root = hook_.topology();
    if (!root.name.empty())
        flatten_node(root, 0);
}

// ── 1. Model Topology ─────────────────────────────────────────────────────────

Element TuiApp::render_topology() {
    rebuild_tree();

    Elements rows;
    for (int i = 0; i < (int)flat_.size(); ++i) {
        const auto& fn = flat_[i];
        std::string pad(fn.depth * 2, ' ');
        bool is_collapsed = collapsed_.count(fn.full_name) > 0;
        std::string bullet = fn.has_children ? (is_collapsed ? "▶ " : "▼ ") : "● ";
        std::string label  = pad + bullet + fn.full_name;

        Color c = Color::White;
        switch (fn.type) {
            case LayerType::Attention:  c = Color::Cyan;    break;
            case LayerType::MLP:        c = Color::Green;   break;
            case LayerType::RMSNorm:    c = Color::Yellow;  break;
            case LayerType::Embedding:  c = Color::Magenta; break;
            case LayerType::Output:     c = Color::Red;     break;
            default: break;
        }

        auto row = text(label) | color(c);
        if (fn.is_capture_target) row = row | bold;
        if (i == topo_cursor_)    row = row | inverted;
        rows.push_back(row);
    }

    if (flat_.empty())
        rows.push_back(text("  (waiting for model…)") | color(Color::GrayDark) | dim);

    bool active = (focus_ == 0);
    auto title = text(active ? "█ 1. MODEL TOPOLOGY" : "  1. MODEL TOPOLOGY")
                 | (active ? color(Color::Cyan) | bold : color(Color::GrayDark));

    return window(title,
        vbox(std::move(rows)) | vscroll_indicator | frame
    ) | flex;
}

// ── 2. Live Packet Stream ─────────────────────────────────────────────────────

Element TuiApp::render_stream() {
    std::vector<TelemetryPacket> pkts;
    pkts.reserve(PACKET_BUF_CAP);
    hook_.packets().for_each([&](const TelemetryPacket& p) { pkts.push_back(p); });

    Elements rows;
    rows.push_back(
        hbox({
            text(" ID  ") | bold | color(Color::GrayLight),
            text("│ TIMESTAMP    ") | bold | color(Color::GrayLight),
            text("│ LAYER TYPE    ") | bold | color(Color::GrayLight),
            text("│ DEVICE") | bold | color(Color::GrayLight),
        })
    );
    rows.push_back(separator());

    // Show newest 20 entries, stream_scroll_ offsets from tail
    int tail = std::max(0, (int)pkts.size() - 20 + stream_scroll_);
    int cnt  = 0;
    for (int i = tail; i < (int)pkts.size() && cnt < 20; ++i, ++cnt) {
        const auto& p = pkts[i];

        Color tc;
        switch (p.layer_type) {
            case LayerType::Attention:  tc = Color::Cyan;    break;
            case LayerType::MLP:        tc = Color::Green;   break;
            case LayerType::RMSNorm:    tc = Color::Yellow;  break;
            case LayerType::Embedding:  tc = Color::Magenta; break;
            case LayerType::Output:     tc = Color::Red;     break;
            default:                    tc = Color::White;   break;
        }

        std::string id_s = std::to_string(p.id);
        while (id_s.size() < 4) id_s = " " + id_s;

        rows.push_back(hbox({
            text(" " + id_s + " ")             | color(Color::GrayDark),
            text("│ " + fmt_time(p.timestamp) + " ") | color(Color::GrayLight),
            text("│ " + trunc(layer_type_str(p.layer_type), 13) + "  ") | color(tc),
            text("│ " + std::string(device_str(p.device)))  | color(Color::GrayLight),
        }));
    }

    if (pkts.empty())
        rows.push_back(text("  (no packets yet)") | color(Color::GrayDark) | dim);

    bool active = (focus_ == 1);
    auto title = text(active ? "█ 2. LIVE PACKET STREAM" : "  2. LIVE PACKET STREAM")
                 | (active ? color(Color::Cyan) | bold : color(Color::GrayDark));

    return window(title,
        vbox(std::move(rows)) | vscroll_indicator | frame
    ) | flex_grow;
}

// ── 3. Attention Matrix ───────────────────────────────────────────────────────

Element TuiApp::render_attention() {
    // Use real captured attention when available, else fall back to synthetic.
    AttentionMatrix real_copy;
    AttentionMatrix* attn = &synth_attn_;
    std::vector<std::string>* toks = &synth_tokens_;

    if (hook_.has_real_attention()) {
        real_copy = hook_.copy_attention();
        attn = &real_copy;
    }

    int nh   = attn->num_heads;
    int seq  = attn->seq_len;
    int head = attn_head_ % nh;
    int vdim = std::min(seq, 8);
    int px   = std::min(attn_pan_x_, std::max(0, seq - vdim));
    int py   = std::min(attn_pan_y_, std::max(0, seq - vdim));

    Elements mat_rows;

    // Column header (token labels)
    {
        Elements hdr;
        hdr.push_back(text("         "));  // row-label space
        for (int c = px; c < px + vdim && c < seq; ++c) {
            std::string tok = (c < (int)toks->size()) ? (*toks)[c] : "t" + std::to_string(c);
            tok = trunc(tok, 6);
            while ((int)tok.size() < 7) tok += " ";
            hdr.push_back(text(tok) | color(Color::GrayLight));
        }
        mat_rows.push_back(hbox(std::move(hdr)));
    }

    // Matrix rows
    for (int r = py; r < py + vdim && r < seq; ++r) {
        Elements row_e;
        std::string rlabel = (r < (int)toks->size()) ? (*toks)[r] : "t" + std::to_string(r);
        rlabel = trunc(rlabel, 7);
        while ((int)rlabel.size() < 8) rlabel += " ";
        row_e.push_back(text(rlabel) | color(Color::GrayLight));

        for (int c = px; c < px + vdim && c < seq; ++c) {
            float v = std::min(1.0f, attn->get(head, r, c) * attn_contrast_);
            auto cell = text(attn_glyph(v)) | color(attn_color(v));
            row_e.push_back(cell);
        }
        mat_rows.push_back(hbox(std::move(row_e)));
    }

    // Controls line
    std::string head_label = "HEAD " + std::to_string(head) + "/" + std::to_string(nh - 1);
    std::string contrast_s = "×" + std::to_string(attn_contrast_).substr(0, 3);
    auto hint = hbox({
        text(" " + head_label) | color(Color::Yellow) | bold,
        text("  [</>]:Head  [hjkl]:Pan  [+/-]:Contrast  ") | color(Color::GrayDark),
        text(contrast_s) | color(Color::Cyan),
        filler(),
        text("Viewport [" + std::to_string(px) + "-" + std::to_string(px+vdim-1) +
             "] × [" + std::to_string(py) + "-" + std::to_string(py+vdim-1) + "] ") | color(Color::GrayDark),
    });

    bool active = (focus_ == 2);
    auto title = text(active ? "█ 3. ATTENTION MATRIX (HEAD " + std::to_string(head) + ")"
                             : "  3. ATTENTION MATRIX (HEAD " + std::to_string(head) + ")")
                 | (active ? color(Color::Cyan) | bold : color(Color::GrayDark));

    return window(title, vbox({
        vbox(std::move(mat_rows)),
        separator(),
        hint,
    }));
}

// ── 4. Runtime Metrics ────────────────────────────────────────────────────────

Element TuiApp::render_metrics() {
    // Find the most-recent packet matching the selected topology node, else overall latest
    TelemetryPacket latest;
    bool has_data = false;

    std::string sel_name;
    if (!flat_.empty() && topo_cursor_ < (int)flat_.size())
        sel_name = flat_[topo_cursor_].full_name;

    hook_.packets().for_each([&](const TelemetryPacket& p) {
        has_data = true;
        latest = p;  // keep last (newest)
    });
    // Prefer layer matching topology selection
    if (!sel_name.empty()) {
        hook_.packets().for_each([&](const TelemetryPacket& p) {
            if (p.layer_name.find(sel_name) != std::string::npos ||
                sel_name.find(p.layer_name)  != std::string::npos) {
                latest = p;
                has_data = true;
            }
        });
    }

    Elements rows;
    if (has_data) {
        const auto& s = latest.tensor_stats;

        auto fmt_f = [](float v) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };

        rows.push_back(hbox({ text("Layer  : ") | bold, text(trunc(latest.layer_name,22)) | color(Color::Cyan) }));
        rows.push_back(hbox({ text("Type   : ") | bold, text(layer_type_str(latest.layer_type)) | color(Color::Yellow) }));
        rows.push_back(hbox({ text("Device : ") | bold, text(device_str(latest.device)) | color(Color::White) }));
        rows.push_back(separator());
        rows.push_back(hbox({ text("Shape  : ") | bold, text(s.shape_str()) | color(Color::Yellow) }));
        rows.push_back(hbox({ text("DType  : ") | bold, text(s.dtype) | color(Color::GrayLight) }));
        rows.push_back(separator());
        rows.push_back(hbox({ text("Mean   : ") | bold, text(fmt_f(s.mean)) | color(Color::White) }));
        Color max_c = s.max_val > 6.0f ? Color::Red : Color::White;
        rows.push_back(hbox({ text("Max    : ") | bold, text(fmt_f(s.max_val)) | color(max_c) }));
        rows.push_back(hbox({ text("Min    : ") | bold, text(fmt_f(s.min_val)) | color(Color::White) }));
        rows.push_back(separator());

        // Sparsity bar
        int pct = (int)(s.sparsity * 100);
        rows.push_back(hbox({
            text("Sparse : ") | bold,
            gauge(s.sparsity) | color(pct > 60 ? Color::Green : Color::Yellow) | flex,
            text(" " + std::to_string(pct) + "%") | color(Color::White),
        }));
        rows.push_back(separator());

        // Latency
        std::ostringstream lat_s;
        lat_s << std::fixed << std::setprecision(3) << latest.latency_ms << " ms";
        rows.push_back(hbox({
            text("Latency: ") | bold,
            text(lat_s.str()) | color(latest.latency_ms > 5.0 ? Color::Red : Color::Green),
        }));
    } else {
        rows.push_back(text(" (no data)") | color(Color::GrayDark) | dim);
    }

    bool active = (focus_ == 3);
    auto title = text(active ? "█ 4. RUNTIME METRICS" : "  4. RUNTIME METRICS")
                 | (active ? color(Color::Cyan) | bold : color(Color::GrayDark));

    return window(title, vbox(std::move(rows))) | flex;
}

// ── 5. Anomaly Ledger ─────────────────────────────────────────────────────────

Element TuiApp::render_anomaly() {
    std::vector<AnomalyRecord> recs;
    hook_.anomalies().for_each([&](const AnomalyRecord& a) { recs.push_back(a); });

    Elements rows;
    for (const auto& a : recs) {
        auto icon = a.is_error
            ? text("✖ ") | color(Color::Red)   | bold
            : text("⚠ ") | color(Color::Yellow) | bold;
        rows.push_back(hbox({
            icon,
            text(fmt_time(a.timestamp) + " ") | color(Color::GrayDark),
            paragraph(a.message),
        }));
    }
    if (recs.empty())
        rows.push_back(text("  ✓ no anomalies detected") | color(Color::Green) | dim);

    bool active = (focus_ == 4);
    auto title = text(active ? "█ 5. ANOMALY LEDGER" : "  5. ANOMALY LEDGER")
                 | (active ? color(Color::Cyan) | bold : color(Color::GrayDark));

    return window(title,
        vbox(std::move(rows)) | vscroll_indicator | frame
    ) | flex;
}

// ── Top-level layout ──────────────────────────────────────────────────────────

Element TuiApp::render_layout() {
    size_t pkt_count = hook_.packets().size();
    size_t anm_count = hook_.anomalies().size();

    // Status bar
    static const char* panel_names[] = {
        "TOPOLOGY", "STREAM", "ATTENTION", "METRICS", "ANOMALY"
    };
    auto status = hbox({
        text(" PRISM v0.3 ") | bold | color(Color::Black) | bgcolor(Color::Cyan),
        text("  [Tab]:Cycle Focus  [Q]:Quit  ") | color(Color::GrayLight),
        text("[j/k]:Navigate  [h/l]:Pan  [+/-]:Contrast  [</>]:Head") | color(Color::GrayDark),
        filler(),
        text("Focus:") | color(Color::GrayDark),
        text(std::string(" ") + panel_names[focus_] + " ") | color(Color::Cyan) | bold,
        text("Packets:") | color(Color::GrayDark),
        text(std::to_string(pkt_count) + " ") | color(Color::White),
        text("Anomalies:") | color(Color::GrayDark),
        text(std::to_string(anm_count) + " ") | color(anm_count > 0 ? Color::Yellow : Color::White),
    });

    return vbox({
        status,
        // Row 1: topology (left) + stream (right)
        hbox({
            render_topology(),
            render_stream(),
        }) | flex,
        // Row 2: attention matrix (full width)
        render_attention() | size(HEIGHT, LESS_THAN, 14),
        // Row 3: metrics (left) + anomaly (right)
        hbox({
            render_metrics(),
            render_anomaly(),
        }) | flex,
    });
}

// ── Event handler ─────────────────────────────────────────────────────────────

bool TuiApp::on_event(Event e) {
    if (e == Event::Character('q') || e == Event::Character('Q')) {
        quit_ = true;
        return true;
    }
    if (e == Event::Tab) {
        focus_ = (focus_ + 1) % 5;
        return true;
    }
    if (e == Event::TabReverse) {
        focus_ = (focus_ + 4) % 5;
        return true;
    }

    switch (focus_) {
        case 0:  // Topology — j/k navigate, Space to expand/collapse
            if (e == Event::Character('j') || e == Event::ArrowDown) {
                topo_cursor_ = std::min((int)flat_.size() - 1, topo_cursor_ + 1);
                return true;
            }
            if (e == Event::Character('k') || e == Event::ArrowUp) {
                topo_cursor_ = std::max(0, topo_cursor_ - 1);
                return true;
            }
            if (e == Event::Character(' ') || e == Event::Return) {
                if (!flat_.empty() && topo_cursor_ < (int)flat_.size()) {
                    const auto& fn = flat_[topo_cursor_];
                    if (fn.has_children) {
                        if (collapsed_.count(fn.full_name)) collapsed_.erase(fn.full_name);
                        else                                 collapsed_.insert(fn.full_name);
                        rebuild_tree();
                        topo_cursor_ = std::min(topo_cursor_, (int)flat_.size() - 1);
                    }
                }
                return true;
            }
            break;

        case 1:  // Stream — j/k to scroll
            if (e == Event::Character('j') || e == Event::ArrowDown) {
                ++stream_scroll_;
                return true;
            }
            if (e == Event::Character('k') || e == Event::ArrowUp) {
                stream_scroll_ = std::max(0, stream_scroll_ - 1);
                return true;
            }
            break;

        case 2:  // Attention — hjkl pan, +/- contrast, </> head
            if (e == Event::Character('l') || e == Event::ArrowRight) { ++attn_pan_x_;                                      return true; }
            if (e == Event::Character('h') || e == Event::ArrowLeft)  { attn_pan_x_ = std::max(0, attn_pan_x_ - 1);        return true; }
            if (e == Event::Character('j') || e == Event::ArrowDown)  { ++attn_pan_y_;                                      return true; }
            if (e == Event::Character('k') || e == Event::ArrowUp)    { attn_pan_y_ = std::max(0, attn_pan_y_ - 1);        return true; }
            if (e == Event::Character('+') || e == Event::Character('=')) { attn_contrast_ = std::min(5.0f, attn_contrast_ + 0.1f); return true; }
            if (e == Event::Character('-'))                            { attn_contrast_ = std::max(0.1f, attn_contrast_ - 0.1f); return true; }
            if (e == Event::Character('>') || e == Event::Character('.')) { ++attn_head_; return true; }
            if (e == Event::Character('<') || e == Event::Character(',')) { attn_head_ = std::max(0, attn_head_ - 1); return true; }
            break;

        default:
            break;
    }
    return false;
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void TuiApp::run() {
    auto screen = ScreenInteractive::Fullscreen();

    auto renderer = Renderer([this]() { return render_layout(); });

    auto root = CatchEvent(renderer, [this, &screen](Event e) -> bool {
        bool handled = on_event(e);
        if (quit_) screen.ExitLoopClosure()();
        return handled;
    });

    // Refresh at ~10 Hz so live packet data animates smoothly
    std::thread refresher([&]() {
        while (!quit_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(root);
    quit_ = true;
    refresher.join();
}
