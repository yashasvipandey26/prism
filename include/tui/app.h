#pragma once

#include "core/hook.h"

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <string>
#include <vector>

// A flattened entry from the ModelNode tree used by the topology panel.
struct FlatNode {
    std::string full_name;
    LayerType   type;
    int         depth;
    bool        has_children;
    bool        is_capture_target;
};

class TuiApp {
public:
    explicit TuiApp(ModelHook& hook);
    void run();

private:
    ModelHook&        hook_;
    std::atomic<bool> quit_{false};

    // 0=topology  1=stream  2=attention  3=metrics  4=anomaly
    int focus_ = 0;

    // Topology panel
    int                  topo_cursor_ = 0;
    std::vector<FlatNode> flat_;

    // Attention panel
    int   attn_head_     = 0;
    int   attn_pan_x_    = 0;
    int   attn_pan_y_    = 0;
    float attn_contrast_ = 1.0f;
    AttentionMatrix          synth_attn_;
    std::vector<std::string> synth_tokens_;

    // Stream panel
    int stream_scroll_ = 0;

    // Helpers
    void rebuild_tree();
    void flatten_node(const ModelNode& node, int depth);

    // Panel renderers
    ftxui::Element render_topology();
    ftxui::Element render_stream();
    ftxui::Element render_attention();
    ftxui::Element render_metrics();
    ftxui::Element render_anomaly();
    ftxui::Element render_layout();

    bool on_event(ftxui::Event e);
};
