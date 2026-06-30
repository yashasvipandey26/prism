#pragma once

#include "core/hook.h"
#include <string>

// ── Trace file format (line-oriented text) ────────────────────────────────────
//
// PRISM_TRACE v1
// META key=value
// PKT  <id> <ts_us> <type> <device> <lat_ms> | <name> | <shape> | <dtype> | <mean> <max> <min> <sparse>
// ANO  <ts_us> <is_err> | <message>
//
// Usage:
//   prism_trace::save(hook, "run.prism")   // after inference
//   prism_trace::load(hook, "run.prism")   // replay in TUI

namespace prism_trace {
    bool save(const ModelHook& hook, const std::string& path);
    bool load(ModelHook& hook, const std::string& path);
}
