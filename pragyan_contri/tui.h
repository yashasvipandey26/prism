#pragma once
#include "instrumentation.h"
#include <ncurses.h>
#include <vector>
#include <string>
#include <functional>

// ─── Panel identifiers ────────────────────────────────────────────────────────
enum class Panel {
    Topology    = 0,
    PacketStream = 1,
    Attention   = 2,
    Metrics     = 3,
    Anomalies   = 4,
    COUNT
};

// ─── TUI ─────────────────────────────────────────────────────────────────────
class TUI {
public:
    TUI();
    ~TUI();

    // Main blocking event loop.
    void run();

private:
    // ── Layout ─────────────────────────────────────────────────────────────
    void recalc_layout();
    void draw_all();

    void draw_statusbar();
    void draw_topology(WINDOW* w, bool focused);
    void draw_packet_stream(WINDOW* w, bool focused);
    void draw_attention(WINDOW* w, bool focused);
    void draw_metrics(WINDOW* w, bool focused);
    void draw_anomalies(WINDOW* w, bool focused);

    // ── Input ──────────────────────────────────────────────────────────────
    void handle_key(int ch);
    void handle_topology_key(int ch);
    void handle_attention_key(int ch);
    void handle_packet_key(int ch);

    // ── Helpers ────────────────────────────────────────────────────────────
    std::string layer_type_str(LayerType t);
    std::string device_str(const LayerRecord& r);
    std::string timestamp_str(const LayerRecord& r);
    std::string sparsity_bar(float s, int width = 13);
    int  layer_color(LayerType t);
    void draw_border_title(WINDOW* w, const std::string& title,
                           int panel_num, bool focused);
    void draw_attention_fullscreen();
    void exit_fullscreen();

    // ── Windows ────────────────────────────────────────────────────────────
    WINDOW* win_topo_   = nullptr;
    WINDOW* win_stream_ = nullptr;
    WINDOW* win_attn_   = nullptr;
    WINDOW* win_metrics_= nullptr;
    WINDOW* win_anom_   = nullptr;
    WINDOW* win_status_ = nullptr;
    WINDOW* win_full_   = nullptr;   // fullscreen overlay

    // Terminal dimensions.
    int rows_ = 0, cols_ = 0;

    // ── State ──────────────────────────────────────────────────────────────
    Panel focus_  = Panel::Topology;
    bool  running_= true;
    bool  fullscreen_attn_ = false;

    // Topology navigation.
    int  topo_cursor_ = 0;
    std::vector<std::pair<int,std::string>> flat_topo_;   // (depth, name)
    int  topo_scroll_  = 0;
    void rebuild_flat_topo();
    void rebuild_flat_topo_rec(const TopoNode& node);

    // Packet stream scroll.
    int  stream_scroll_ = 0;
    bool stream_follow_ = true;

    // Attention pan.
    int attn_head_  = 0;
    int attn_pan_r_ = 0;
    int attn_pan_c_ = 0;

    // Selected layer (topology cursor → LayerRecord cache).
    int selected_layer_idx_ = -1;   // index into last snapshot

    // Contrast multiplier for attention matrix.
    float attn_contrast_ = 1.f;

    // Cache of last snapshot.
    std::vector<LayerRecord> snap_;
    void refresh_snap();
};
