#include "core/trace_io.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ── Helpers ───────────────────────────────────────────────────────────────────

static int64_t to_us(std::chrono::system_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               tp.time_since_epoch()).count();
}

static std::chrono::system_clock::time_point from_us(int64_t us) {
    return std::chrono::system_clock::time_point(std::chrono::microseconds(us));
}

// Encode a string so it survives a single-line format: replace '|' and '\n'.
static std::string encode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '|')  { out += "\\p"; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else                { out += c; }
    }
    return out;
}

static std::string decode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case 'p':  out += '|'; ++i; break;
                case '\\': out += '\\'; ++i; break;
                case 'n':  out += '\n'; ++i; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

// Serialize a vector<int64_t> shape to "[1, 32, 4096]"
static std::string shape_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

// Parse "[1, 32, 4096]" back to vector<int64_t>
static std::vector<int64_t> parse_shape(const std::string& s) {
    std::vector<int64_t> out;
    bool in_num = false;
    int64_t cur = 0;
    bool neg = false;
    for (char c : s) {
        if (c == '-') { neg = true; in_num = true; }
        else if (c >= '0' && c <= '9') { cur = cur * 10 + (c - '0'); in_num = true; }
        else if (in_num) { out.push_back(neg ? -cur : cur); cur = 0; neg = false; in_num = false; }
    }
    if (in_num) out.push_back(neg ? -cur : cur);
    return out;
}

// Split a string at the first occurrence of sep.
static std::pair<std::string, std::string> split1(const std::string& s, char sep) {
    auto pos = s.find(sep);
    if (pos == std::string::npos) return {s, ""};
    return {s.substr(0, pos), s.substr(pos + 1)};
}

// ── Save ──────────────────────────────────────────────────────────────────────

bool prism_trace::save(const ModelHook& hook, const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;

    f << "PRISM_TRACE v1\n";

    const ModelNode& root = hook.topology();
    if (!root.name.empty())
        f << "META model=" << encode(root.name) << "\n";
    f << "META saved_at=" << to_us(std::chrono::system_clock::now()) << "\n";
    f << "\n";

    // Packets
    hook.packets().for_each([&](const TelemetryPacket& p) {
        f << "PKT"
          << " " << p.id
          << " " << to_us(p.timestamp)
          << " " << (int)p.layer_type
          << " " << (int)p.device
          << " " << p.latency_ms
          << " | " << encode(p.layer_name)
          << " | " << shape_str(p.tensor_stats.shape)
          << " | " << encode(p.tensor_stats.dtype)
          << " | " << p.tensor_stats.mean
          << " "   << p.tensor_stats.max_val
          << " "   << p.tensor_stats.min_val
          << " "   << p.tensor_stats.sparsity
          << "\n";
    });

    // Anomalies
    hook.anomalies().for_each([&](const AnomalyRecord& a) {
        f << "ANO"
          << " " << to_us(a.timestamp)
          << " " << (int)a.is_error
          << " | " << encode(a.message)
          << "\n";
    });

    return f.good();
}

// ── Load ──────────────────────────────────────────────────────────────────────

bool prism_trace::load(ModelHook& hook, const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;

    std::string line;
    if (!std::getline(f, line) || line.find("PRISM_TRACE") == std::string::npos)
        return false;  // not a prism trace

    // Reconstruct a synthetic topology stub
    ModelNode root;
    root.name = "(replayed)";
    root.expanded = true;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line.rfind("META ", 0) == 0) {
            auto [key, val] = split1(line.substr(5), '=');
            if (key == "model") root.name = decode(val) + " [replay]";
            continue;
        }

        if (line.rfind("PKT ", 0) == 0) {
            // PKT id ts_us type device lat | name | shape | dtype | mean max min sparse
            std::istringstream ss(line.substr(4));
            TelemetryPacket pkt;
            int64_t ts_us;
            int type_i, dev_i;
            ss >> pkt.id >> ts_us >> type_i >> dev_i >> pkt.latency_ms;
            pkt.timestamp  = from_us(ts_us);
            pkt.layer_type = (LayerType)type_i;
            pkt.device     = (ComputeDevice)dev_i;

            std::string rest;
            std::getline(ss, rest);  // " | name | shape | dtype | stats"
            // Split on '|' — there are 4 '|' separated fields after the spaces
            auto [seg0, rem0] = split1(rest, '|');
            auto [seg1, rem1] = split1(rem0, '|');
            auto [seg2, rem2] = split1(rem1, '|');
            auto [seg3, rem3] = split1(rem2, '|');

            // seg0 is leading whitespace; actual fields are seg1..seg3 + rem3
            pkt.layer_name = decode(seg1.size() > 1 ? seg1.substr(1) : seg1);
            // trim leading space
            if (!pkt.layer_name.empty() && pkt.layer_name.front() == ' ')
                pkt.layer_name = pkt.layer_name.substr(1);

            std::string shape_s = seg2;
            if (!shape_s.empty() && shape_s.front() == ' ') shape_s = shape_s.substr(1);
            pkt.tensor_stats.shape = parse_shape(shape_s);

            std::string dtype_s = seg3;
            if (!dtype_s.empty() && dtype_s.front() == ' ') dtype_s = dtype_s.substr(1);
            pkt.tensor_stats.dtype = decode(dtype_s);

            std::istringstream stats_ss(rem3);
            stats_ss >> pkt.tensor_stats.mean >> pkt.tensor_stats.max_val
                     >> pkt.tensor_stats.min_val >> pkt.tensor_stats.sparsity;

            hook.packets().push(pkt);

            // Add topology node for first occurrence of this layer
            ModelNode n;
            n.name = pkt.layer_name;
            n.type = pkt.layer_type;
            root.children.push_back(n);
            continue;
        }

        if (line.rfind("ANO ", 0) == 0) {
            std::istringstream ss(line.substr(4));
            AnomalyRecord rec;
            int64_t ts_us; int is_err;
            ss >> ts_us >> is_err;
            rec.timestamp = from_us(ts_us);
            rec.is_error  = is_err != 0;

            std::string rest;
            std::getline(ss, rest);
            auto [_, msg] = split1(rest, '|');
            if (!msg.empty() && msg.front() == ' ') msg = msg.substr(1);
            rec.message = decode(msg);
            hook.anomalies().push(rec);
            continue;
        }
    }

    hook.set_topology(std::move(root));
    return true;
}
