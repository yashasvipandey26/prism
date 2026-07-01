#include "tui.h"
#include <ncurses.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <thread>

// ─── Color pair IDs ──────────────────────────────────────────────────────────
enum Color {
    C_NORMAL   = 1,
    C_TITLE    = 2,
    C_FOCUS    = 3,
    C_ATTN     = 4,
    C_MLP      = 5,
    C_NORM     = 6,
    C_EMBED    = 7,
    C_WARN     = 8,
    C_ERR      = 9,
    C_DIM      = 10,
    C_HEADER   = 11,
    C_SELECTED = 12,
    C_BAR_HI   = 13,
    C_BAR_MED  = 14,
    C_BAR_LO   = 15,
    C_CUDA     = 16,
    C_CPU      = 17,
    C_STATUS   = 18,
};

// ─── Block chars for attention heatmap ───────────────────────────────────────
static const char* HEAT_CHARS[] = {" ", "░", "▒", "▓", "█"};

// ─── Constructor / Destructor ─────────────────────────────────────────────────

TUI::TUI() {
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    nodelay(stdscr, TRUE);  // non-blocking getch
    set_escdelay(25);

    start_color();
    use_default_colors();

    // Define color pairs.
    init_pair(C_NORMAL,   COLOR_WHITE,   -1);
    init_pair(C_TITLE,    COLOR_CYAN,    -1);
    init_pair(C_FOCUS,    COLOR_BLACK,   COLOR_CYAN);
    init_pair(C_ATTN,     COLOR_YELLOW,  -1);
    init_pair(C_MLP,      COLOR_GREEN,   -1);
    init_pair(C_NORM,     COLOR_MAGENTA, -1);
    init_pair(C_EMBED,    COLOR_BLUE,    -1);
    init_pair(C_WARN,     COLOR_YELLOW,  -1);
    init_pair(C_ERR,      COLOR_RED,     -1);
    init_pair(C_DIM,      COLOR_WHITE,   -1);
    init_pair(C_HEADER,   COLOR_BLACK,   COLOR_WHITE);
    init_pair(C_SELECTED, COLOR_BLACK,   COLOR_YELLOW);
    init_pair(C_BAR_HI,   COLOR_GREEN,   -1);
    init_pair(C_BAR_MED,  COLOR_YELLOW,  -1);
    init_pair(C_BAR_LO,   COLOR_RED,     -1);
    init_pair(C_CUDA,     COLOR_GREEN,   -1);
    init_pair(C_CPU,      COLOR_RED,     -1);
    init_pair(C_STATUS,   COLOR_BLACK,   COLOR_BLUE);

    recalc_layout();
    rebuild_flat_topo();
}

TUI::~TUI() {
    if (win_topo_)    delwin(win_topo_);
    if (win_stream_)  delwin(win_stream_);
    if (win_attn_)    delwin(win_attn_);
    if (win_metrics_) delwin(win_metrics_);
    if (win_anom_)    delwin(win_anom_);
    if (win_status_)  delwin(win_status_);
    if (win_full_)    delwin(win_full_);
    endwin();
}

// ─── Layout ──────────────────────────────────────────────────────────────────

void TUI::recalc_layout() {
    getmaxyx(stdscr, rows_, cols_);

    // Destroy old windows.
    if (win_topo_)    { delwin(win_topo_);    win_topo_    = nullptr; }
    if (win_stream_)  { delwin(win_stream_);  win_stream_  = nullptr; }
    if (win_attn_)    { delwin(win_attn_);    win_attn_    = nullptr; }
    if (win_metrics_) { delwin(win_metrics_); win_metrics_ = nullptr; }
    if (win_anom_)    { delwin(win_anom_);    win_anom_    = nullptr; }
    if (win_status_)  { delwin(win_status_);  win_status_  = nullptr; }

    int status_h = 1;
    int body_rows = rows_ - status_h;

    // Row 0: topology (left 28 cols) | packet stream (rest)  — top 40%
    int top_h    = body_rows * 40 / 100;
    int left_w   = std::min(40, cols_ / 3);
    int right_w  = cols_ - left_w;

    // Row 1: attention matrix — middle 30%
    int mid_h    = body_rows * 32 / 100;

    // Row 2: metrics (left half) | anomalies (right half) — bottom 28%
    int bot_h    = body_rows - top_h - mid_h;
    int half_w   = cols_ / 2;

    win_topo_    = newwin(top_h, left_w,  0, 0);
    win_stream_  = newwin(top_h, right_w, 0, left_w);
    win_attn_    = newwin(mid_h, cols_,   top_h, 0);
    win_metrics_ = newwin(bot_h, half_w,  top_h + mid_h, 0);
    win_anom_    = newwin(bot_h, cols_ - half_w, top_h + mid_h, half_w);
    win_status_  = newwin(status_h, cols_, rows_ - status_h, 0);

    keypad(win_topo_,   TRUE);
    keypad(win_stream_, TRUE);
    keypad(win_attn_,   TRUE);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

std::string TUI::layer_type_str(LayerType t) {
    switch (t) {
        case LayerType::Attention:  return "Attn (Self) ";
        case LayerType::MLP:        return "MLP (SwiGLU)";
        case LayerType::LayerNorm:  return "LayerNorm   ";
        case LayerType::Embedding:  return "Embedding   ";
        case LayerType::Output:     return "LM Head     ";
        default:                    return "Unknown     ";
    }
}

std::string TUI::device_str(const LayerRecord& r) {
    if (r.is_fallback) return "CPU (Fallback)";
    if (r.device == DeviceType::CUDA)
        return "CUDA [GPU " + std::to_string(r.device_idx) + "]";
    if (r.device == DeviceType::Metal) return "Metal";
    return "CPU";
}

std::string TUI::timestamp_str(const LayerRecord& r) {
    auto t = std::chrono::system_clock::to_time_t(r.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  r.timestamp.time_since_epoch()).count() % 1000;
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    char full[24];
    std::snprintf(full, sizeof(full), "%s.%03lld", buf, (long long)ms);
    return full;
}

std::string TUI::sparsity_bar(float s, int width) {
    int filled = static_cast<int>(s * width);
    std::string bar;
    // Green blocks for dense, yellow for medium, red for very sparse.
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            if (s > 0.7f) bar += "█";
            else if (s > 0.4f) bar += "▓";
            else bar += "░";
        } else {
            bar += "░";
        }
    }
    return bar;
}

int TUI::layer_color(LayerType t) {
    switch (t) {
        case LayerType::Attention: return C_ATTN;
        case LayerType::MLP:       return C_MLP;
        case LayerType::LayerNorm: return C_NORM;
        case LayerType::Embedding: return C_EMBED;
        default:                   return C_NORMAL;
    }
}

void TUI::draw_border_title(WINDOW* w, const std::string& title,
                             int panel_num, bool focused)
{
    int wh, ww;
    getmaxyx(w, wh, ww);
    (void)wh;

    if (focused) {
        wattron(w, COLOR_PAIR(C_FOCUS) | A_BOLD);
        box(w, 0, 0);
        wattroff(w, COLOR_PAIR(C_FOCUS) | A_BOLD);
    } else {
        wattron(w, COLOR_PAIR(C_DIM));
        box(w, 0, 0);
        wattroff(w, COLOR_PAIR(C_DIM));
    }

    std::string hdr = " " + std::to_string(panel_num) + ". " + title + " ";
    int hdr_x = 2;
    if (focused)
        wattron(w, COLOR_PAIR(C_FOCUS) | A_BOLD);
    else
        wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, 0, hdr_x, "%s", hdr.c_str());
    if (focused)
        wattroff(w, COLOR_PAIR(C_FOCUS) | A_BOLD);
    else
        wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    (void)ww;
}

// ─── Topology flat list ───────────────────────────────────────────────────────

void TUI::rebuild_flat_topo_rec(const TopoNode& node) {
    flat_topo_.push_back({node.depth, node.name});
    if (node.expanded) {
        for (auto& c : node.children)
            rebuild_flat_topo_rec(c);
    }
}

void TUI::rebuild_flat_topo() {
    flat_topo_.clear();
    auto& root = InstrumentationEngine::instance().topology();
    rebuild_flat_topo_rec(root);
}

// ─── Snapshot refresh ─────────────────────────────────────────────────────────

void TUI::refresh_snap() {
    snap_ = InstrumentationEngine::instance().buffer().snapshot();
}

// ─── Status bar ───────────────────────────────────────────────────────────────

void TUI::draw_statusbar() {
    wclear(win_status_);
    wattron(win_status_, COLOR_PAIR(C_STATUS) | A_BOLD);
    // Fill entire line.
    for (int i = 0; i < cols_; ++i) mvwaddch(win_status_, 0, i, ' ');

    auto& eng = InstrumentationEngine::instance();
    std::string model = eng.model_name();
    bool sim = eng.is_simulating();

    std::string status =
        " [Tab]:Focus  [j/k]:Nav  [Space]:Expand  [h/l/←/→]:Attn  [F]:Fullscreen  "
        "[+/-]:Contrast  [Q]:Quit ";
    if (sim) status = " ● LIVE  " + status;

    mvwprintw(win_status_, 0, 0, "%s", status.substr(0, cols_-1).c_str());

    // Right-align model name.
    std::string mstr = " " + model + " ";
    int mx = cols_ - static_cast<int>(mstr.size());
    if (mx > 0) mvwprintw(win_status_, 0, mx, "%s", mstr.c_str());

    wattroff(win_status_, COLOR_PAIR(C_STATUS) | A_BOLD);
    wrefresh(win_status_);
}

// ─── Panel 1: Topology ────────────────────────────────────────────────────────

void TUI::draw_topology(WINDOW* w, bool focused) {
    wclear(w);
    draw_border_title(w, "MODEL TOPOLOGY", 1, focused);

    int wh, ww;
    getmaxyx(w, wh, ww);

    rebuild_flat_topo();

    int visible = wh - 3;
    if (topo_cursor_ < topo_scroll_) topo_scroll_ = topo_cursor_;
    if (topo_cursor_ >= topo_scroll_ + visible)
        topo_scroll_ = topo_cursor_ - visible + 1;

    for (int i = 0; i < visible; ++i) {
        int idx = i + topo_scroll_;
        if (idx >= (int)flat_topo_.size()) break;
        auto& [depth, name] = flat_topo_[idx];

        bool is_cursor = (idx == topo_cursor_);
        int row = i + 1;

        if (is_cursor) {
            wattron(w, COLOR_PAIR(C_SELECTED) | A_BOLD);
        }

        // Indent.
        std::string indent(depth * 2, ' ');
        std::string prefix;
        if (depth == 0)      prefix = "▼ ";
        else if (depth == 1) prefix = "► ";
        else if (depth == 2) prefix = "▼ ";
        else                 prefix = "● ";

        std::string line = indent + prefix + name;
        if ((int)line.size() >= ww - 2)
            line = line.substr(0, ww - 3) + "…";

        mvwprintw(w, row, 1, "%-*s", ww - 2, line.c_str());

        if (is_cursor) wattroff(w, COLOR_PAIR(C_SELECTED) | A_BOLD);
    }

    // Footer hint.
    wattron(w, COLOR_PAIR(C_DIM) | A_DIM);
    std::string hint = " [j/k]:Nav [Space]:Expand ";
    mvwprintw(w, wh - 1, 1, "%s", hint.substr(0, ww - 2).c_str());
    wattroff(w, COLOR_PAIR(C_DIM) | A_DIM);

    wrefresh(w);
}

// ─── Panel 2: Packet Stream ───────────────────────────────────────────────────

void TUI::draw_packet_stream(WINDOW* w, bool focused) {
    wclear(w);
    draw_border_title(w, "LIVE PACKET STREAM", 2, focused);

    int wh, ww;
    getmaxyx(w, wh, ww);

    // Column widths.
    //  ID(6) | TS(14) | TYPE(14) | DEVICE(18) | LAT(10)
    const int id_w   = 6;
    const int ts_w   = 14;
    const int type_w = 14;
    const int dev_w  = 18;
    const int lat_w  = 10;

    // Header row.
    wattron(w, COLOR_PAIR(C_HEADER) | A_BOLD);
    mvwprintw(w, 1, 1, " %-*s %-*s %-*s %-*s %-*s",
              id_w,   "ID",
              ts_w,   "TIMESTAMP",
              type_w, "LAYER TYPE",
              dev_w,  "DEVICE",
              lat_w,  "LATENCY");
    // Fill rest of header.
    for (int x = 1 + id_w + ts_w + type_w + dev_w + lat_w + 5; x < ww - 1; ++x)
        mvwaddch(w, 1, x, ' ');
    wattroff(w, COLOR_PAIR(C_HEADER) | A_BOLD);

    int visible = wh - 4;
    if (visible <= 0) { wrefresh(w); return; }

    if (snap_.empty()) { wrefresh(w); return; }

    // Clamp scroll.
    int max_scroll = std::max(0, (int)snap_.size() - visible);
    if (stream_follow_) stream_scroll_ = max_scroll;
    stream_scroll_ = std::max(0, std::min(stream_scroll_, max_scroll));

    for (int i = 0; i < visible; ++i) {
        int idx = stream_scroll_ + i;
        if (idx >= (int)snap_.size()) break;
        auto& rec = snap_[idx];

        int row = i + 2;
        bool is_sel = (idx == selected_layer_idx_);

        // Row background.
        if (is_sel) wattron(w, A_REVERSE);

        // ID.
        wattron(w, COLOR_PAIR(C_DIM));
        mvwprintw(w, row, 1, " %*llu ", id_w - 1, (unsigned long long)rec.id);
        wattroff(w, COLOR_PAIR(C_DIM));

        // Timestamp.
        std::string ts = timestamp_str(rec);
        mvwprintw(w, row, 1 + id_w + 1, "%-*s ", ts_w, ts.substr(0, ts_w).c_str());

        // Type with color.
        wattron(w, COLOR_PAIR(layer_color(rec.type)) | A_BOLD);
        std::string lt = layer_type_str(rec.type);
        mvwprintw(w, row, 1 + id_w + ts_w + 2, "%-*s ", type_w, lt.c_str());
        wattroff(w, COLOR_PAIR(layer_color(rec.type)) | A_BOLD);

        // Device.
        std::string dev = device_str(rec);
        int dcol = rec.is_fallback ? C_CPU :
                   (rec.device == DeviceType::CUDA ? C_CUDA : C_CPU);
        wattron(w, COLOR_PAIR(dcol));
        mvwprintw(w, row, 1 + id_w + ts_w + type_w + 3,
                  "%-*s ", dev_w, dev.substr(0, dev_w).c_str());
        wattroff(w, COLOR_PAIR(dcol));

        // Latency.
        char lat_buf[16];
        std::snprintf(lat_buf, sizeof(lat_buf), "%.3f ms", rec.latency_ms);
        int lat_col = rec.latency_ms > 4.0 ? C_WARN :
                      rec.latency_ms > 2.0 ? C_BAR_MED : C_BAR_HI;
        wattron(w, COLOR_PAIR(lat_col));
        mvwprintw(w, row, 1 + id_w + ts_w + type_w + dev_w + 4,
                  "%-*s", lat_w, lat_buf);
        wattroff(w, COLOR_PAIR(lat_col));

        if (is_sel) wattroff(w, A_REVERSE);
    }

    // Follow indicator.
    wattron(w, COLOR_PAIR(C_DIM) | A_DIM);
    std::string follow_str = stream_follow_ ? " [↓ FOLLOW] " : " [PAUSED] ";
    mvwprintw(w, wh - 1, 1, "%s", follow_str.c_str());
    wattroff(w, COLOR_PAIR(C_DIM) | A_DIM);

    wrefresh(w);
}

// ─── Panel 3: Attention Matrix ────────────────────────────────────────────────

static std::string heat_char(float v) {
    if (v < 0.05f) return HEAT_CHARS[0];
    if (v < 0.20f) return HEAT_CHARS[1];
    if (v < 0.40f) return HEAT_CHARS[2];
    if (v < 0.65f) return HEAT_CHARS[3];
    return HEAT_CHARS[4];
}

void TUI::draw_attention(WINDOW* w, bool focused) {
    wclear(w);
    int wh, ww;
    getmaxyx(w, wh, ww);

    std::string head_title = "ATTENTION MATRIX VISUALIZER (HEAD " +
                             std::to_string(attn_head_) + ")";
    draw_border_title(w, head_title, 3, focused);

    // Find latest record with attention data.
    const AttentionSnapshot* snap = nullptr;
    const LayerRecord* src = nullptr;
    for (int i = (int)snap_.size() - 1; i >= 0; --i) {
        if (snap_[i].attn.has_value()) {
            snap = &snap_[i].attn.value();
            src  = &snap_[i];
            break;
        }
    }

    if (!snap || snap->attn_weights.empty()) {
        wattron(w, COLOR_PAIR(C_DIM) | A_DIM);
        mvwprintw(w, wh / 2, ww / 2 - 15, "Waiting for attention data...");
        wattroff(w, COLOR_PAIR(C_DIM) | A_DIM);
        wrefresh(w);
        return;
    }

    int num_heads = static_cast<int>(snap->attn_weights.size());
    attn_head_ = std::min(attn_head_, num_heads - 1);
    auto& mat = snap->attn_weights[attn_head_];

    int seq = static_cast<int>(mat.size());

    // Token header row.
    // Each cell is 2 chars wide.
    int cell_w = 2;
    int label_col_w = 9;  // left label column.
    int max_display_cols = (ww - label_col_w - 2) / cell_w;
    int max_display_rows = wh - 4;

    int col_start = std::min(attn_pan_c_, std::max(0, seq - max_display_cols));
    int row_start = std::min(attn_pan_r_, std::max(0, seq - max_display_rows));
    col_start = std::max(0, col_start);
    row_start = std::max(0, row_start);
    int col_end = std::min(seq, col_start + max_display_cols);
    int row_end = std::min(seq, row_start + max_display_rows);

    // Token labels header.
    wattron(w, COLOR_PAIR(C_DIM) | A_BOLD);
    mvwprintw(w, 1, 1, "Tokens: ");
    wattroff(w, COLOR_PAIR(C_DIM) | A_BOLD);

    int tx = 9;
    for (int c = col_start; c < col_end && tx < ww - 4; ++c) {
        std::string tok = (c < (int)snap->tokens.size()) ? snap->tokens[c] : "?";
        if (tok.size() > 1) tok = tok.substr(0, 1);
        wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
        mvwprintw(w, 1, tx, "[%s]", tok.c_str());
        wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);
        tx += cell_w + 2;
    }

    // Matrix rows.
    for (int r = row_start; r < row_end; ++r) {
        int wy = r - row_start + 2;
        if (wy >= wh - 1) break;

        // Row label.
        std::string row_tok = (r < (int)snap->tokens.size()) ? snap->tokens[r] : "?";
        if (row_tok.size() > 6) row_tok = row_tok.substr(0, 6);
        wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
        mvwprintw(w, wy, 1, "[%-6s]", row_tok.c_str());
        wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);

        // Cells.
        int cx = label_col_w;
        for (int c = col_start; c < col_end; ++c) {
            float val = (r < (int)mat.size() && c < (int)mat[r].size())
                        ? std::min(1.f, mat[r][c] * attn_contrast_) : 0.f;
            int heat_idx = static_cast<int>(val * 4.99f);
            heat_idx = std::max(0, std::min(4, heat_idx));

            // Color: bright = high attention.
            int cpair = (heat_idx >= 4) ? C_BAR_HI :
                        (heat_idx >= 2) ? C_BAR_MED : C_DIM;
            wattron(w, COLOR_PAIR(cpair));
            mvwprintw(w, wy, cx, "%s%s",
                      HEAT_CHARS[heat_idx], HEAT_CHARS[heat_idx]);
            wattroff(w, COLOR_PAIR(cpair));
            cx += cell_w;
            if (cx >= ww - 2) break;
        }
    }

    // Viewport label + controls (right side).
    int ctrl_x = ww - 40;
    if (ctrl_x > label_col_w + (col_end - col_start) * cell_w + 2) {
        wattron(w, COLOR_PAIR(C_DIM) | A_DIM);
        std::string vp = "Viewport: [" + std::to_string(col_start) + "-" +
                         std::to_string(col_end - 1) + "] x [" +
                         std::to_string(row_start) + "-" +
                         std::to_string(row_end - 1) + "]";
        mvwprintw(w, 1, ctrl_x, "%s", vp.c_str());
        mvwprintw(w, 2, ctrl_x, "[F]: Fullscreen");
        mvwprintw(w, 3, ctrl_x, "[h/j/k/l]: Pan");
        mvwprintw(w, 4, ctrl_x, "[+/-]: Contrast x%.1f", attn_contrast_);
        mvwprintw(w, 5, ctrl_x, "[H/L]: Head %d/%d", attn_head_ + 1, num_heads);
        wattroff(w, COLOR_PAIR(C_DIM) | A_DIM);
    }

    wrefresh(w);
}

// ─── Panel 4: Runtime Metrics ─────────────────────────────────────────────────

void TUI::draw_metrics(WINDOW* w, bool focused) {
    wclear(w);
    draw_border_title(w, "RUNTIME METRICS INSPECTOR", 4, focused);

    int wh, ww;
    getmaxyx(w, wh, ww);
    (void)wh;

    // Use selected layer or last record.
    const LayerRecord* rec = nullptr;
    if (selected_layer_idx_ >= 0 && selected_layer_idx_ < (int)snap_.size())
        rec = &snap_[selected_layer_idx_];
    else if (!snap_.empty())
        rec = &snap_.back();

    if (!rec) {
        wattron(w, COLOR_PAIR(C_DIM) | A_DIM);
        mvwprintw(w, 2, 2, "No data yet...");
        wattroff(w, COLOR_PAIR(C_DIM) | A_DIM);
        wrefresh(w);
        return;
    }

    int row = 1;

    // Tensor shape.
    wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, 2, "Tensor Shape : ");
    wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    wattron(w, COLOR_PAIR(C_NORMAL));
    std::string shape_str = rec->output_shape.to_string();
    mvwprintw(w, row, 17, "%-20s", shape_str.c_str());

    wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, std::min(40, ww/2), "  Dtype: ");
    wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, std::min(49, ww/2 + 9), "%s", rec->stats.dtype.c_str());
    wattroff(w, COLOR_PAIR(C_NORMAL));
    ++row;

    // Sparsity.
    wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, 2, "Sparsity Rate: ");
    wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);

    float s = rec->stats.sparsity;
    // Colour-coded bar segments.
    mvwprintw(w, row, 17, "");
    int bar_len = 13;
    int filled  = static_cast<int>(s * bar_len);
    for (int i = 0; i < bar_len; ++i) {
        int cpair = (i < filled)
                    ? (s > 0.6f ? C_BAR_HI : s > 0.3f ? C_BAR_MED : C_BAR_LO)
                    : C_DIM;
        wattron(w, COLOR_PAIR(cpair));
        mvwprintw(w, row, 17 + i, (i < filled) ? "█" : "░");
        wattroff(w, COLOR_PAIR(cpair));
    }
    mvwprintw(w, row, 17 + bar_len + 1, "%.1f%%", s * 100.f);
    ++row;

    // Latency delta.
    wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, 2, "Latency Delta: ");
    wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    double delta = rec->latency_ms - rec->mean_latency_ms;
    int lcol = std::abs(delta) > 2.0 ? C_WARN : C_BAR_HI;
    wattron(w, COLOR_PAIR(lcol));
    mvwprintw(w, row, 17, "%.3f ms  (%s)",
              delta,
              std::abs(delta) < 2.0 ? "Within Normal Bounds" : "ABOVE THRESHOLD");
    wattroff(w, COLOR_PAIR(lcol));
    ++row;

    // Mean & Std dev.
    wattron(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, 2, "Act Stats    : ");
    wattroff(w, COLOR_PAIR(C_TITLE) | A_BOLD);
    mvwprintw(w, row, 17, "mean=%.4f  std=%.4f  min=%.3f  max=%.3f",
              rec->stats.mean, rec->stats.std_dev,
              rec->stats.min_val, rec->stats.max_val);
    ++row;

    // Layer name.
    wattron(w, COLOR_PAIR(C_DIM) | A_DIM);
    mvwprintw(w, row, 2, "Layer: %s  [ID:%llu  lat:%.3fms  mean:%.3fms]",
              rec->name.c_str(),
              (unsigned long long)rec->id,
              rec->latency_ms,
              rec->mean_latency_ms);
    wattroff(w, COLOR_PAIR(C_DIM) | A_DIM);

    wrefresh(w);
}

// ─── Panel 5: Anomaly Ledger ──────────────────────────────────────────────────

void TUI::draw_anomalies(WINDOW* w, bool focused) {
    wclear(w);
    draw_border_title(w, "NUMERICAL ANOMALY LEDGER", 5, focused);

    int wh, ww;
    getmaxyx(w, wh, ww);

    int row = 1;
    for (int i = (int)snap_.size() - 1; i >= 0 && row < wh - 1; --i) {
        auto& rec = snap_[i];
        if (!rec.has_outlier && !rec.is_fallback && !rec.has_nan_inf) continue;

        std::string ts = timestamp_str(rec);
        std::string msg = rec.anomaly_msg;

        // Truncate to window width.
        std::string line = ts + "  " + msg;
        if ((int)line.size() > ww - 3)
            line = line.substr(0, ww - 4) + "…";

        if (rec.has_nan_inf) {
            wattron(w, COLOR_PAIR(C_ERR) | A_BOLD);
            mvwprintw(w, row, 1, "✖ ");
            wattroff(w, COLOR_PAIR(C_ERR) | A_BOLD);
            wattron(w, COLOR_PAIR(C_ERR));
        } else if (rec.is_fallback) {
            wattron(w, COLOR_PAIR(C_ERR) | A_BOLD);
            mvwprintw(w, row, 1, "✖ ");
            wattroff(w, COLOR_PAIR(C_ERR) | A_BOLD);
            wattron(w, COLOR_PAIR(C_WARN));
        } else {
            wattron(w, COLOR_PAIR(C_WARN) | A_BOLD);
            mvwprintw(w, row, 1, "⚠ ");
            wattroff(w, COLOR_PAIR(C_WARN) | A_BOLD);
            wattron(w, COLOR_PAIR(C_WARN));
        }

        mvwprintw(w, row, 3, "%s", line.c_str());
        wattroff(w, COLOR_PAIR(C_WARN));
        wattroff(w, COLOR_PAIR(C_ERR));
        ++row;
    }

    if (row == 1) {
        wattron(w, COLOR_PAIR(C_BAR_HI) | A_DIM);
        mvwprintw(w, 1, 2, "✓ No anomalies detected.");
        wattroff(w, COLOR_PAIR(C_BAR_HI) | A_DIM);
    }

    wrefresh(w);
}

// ─── Fullscreen Attention ─────────────────────────────────────────────────────

void TUI::draw_attention_fullscreen() {
    if (!win_full_) {
        win_full_ = newwin(rows_, cols_, 0, 0);
        keypad(win_full_, TRUE);
    }
    // Delegate to normal draw but in a big window.
    draw_attention(win_full_, true);
}

void TUI::exit_fullscreen() {
    fullscreen_attn_ = false;
    if (win_full_) {
        delwin(win_full_);
        win_full_ = nullptr;
    }
    touchwin(stdscr);
    refresh();
}

// ─── Draw all ─────────────────────────────────────────────────────────────────

void TUI::draw_all() {
    refresh_snap();

    if (fullscreen_attn_) {
        draw_attention_fullscreen();
        draw_statusbar();
        return;
    }

    draw_topology   (win_topo_,    focus_ == Panel::Topology);
    draw_packet_stream(win_stream_, focus_ == Panel::PacketStream);
    draw_attention  (win_attn_,    focus_ == Panel::Attention);
    draw_metrics    (win_metrics_,  focus_ == Panel::Metrics);
    draw_anomalies  (win_anom_,    focus_ == Panel::Anomalies);
    draw_statusbar();
}

// ─── Input handling ───────────────────────────────────────────────────────────

void TUI::handle_topology_key(int ch) {
    int sz = static_cast<int>(flat_topo_.size());
    if (ch == 'j' || ch == KEY_DOWN) {
        topo_cursor_ = std::min(topo_cursor_ + 1, sz - 1);
    } else if (ch == 'k' || ch == KEY_UP) {
        topo_cursor_ = std::max(topo_cursor_ - 1, 0);
    } else if (ch == ' ' || ch == '\n') {
        // Select this layer — find matching record.
        if (topo_cursor_ < sz) {
            auto& [depth, name] = flat_topo_[topo_cursor_];
            for (int i = (int)snap_.size() - 1; i >= 0; --i) {
                if (snap_[i].name.find(name) != std::string::npos) {
                    selected_layer_idx_ = i;
                    break;
                }
            }
        }
    }
}

void TUI::handle_attention_key(int ch) {
    if (ch == 'h' || ch == KEY_LEFT)  attn_pan_c_ = std::max(0, attn_pan_c_ - 1);
    if (ch == 'l' || ch == KEY_RIGHT) attn_pan_c_++;
    if (ch == 'k' || ch == KEY_UP)    attn_pan_r_ = std::max(0, attn_pan_r_ - 1);
    if (ch == 'j' || ch == KEY_DOWN)  attn_pan_r_++;
    if (ch == '+' || ch == '=')       attn_contrast_ = std::min(5.f, attn_contrast_ + 0.25f);
    if (ch == '-')                    attn_contrast_ = std::max(0.25f, attn_contrast_ - 0.25f);
    if (ch == 'H') attn_head_ = std::max(0, attn_head_ - 1);
    if (ch == 'L') attn_head_++;
    if (ch == 'f' || ch == 'F') {
        fullscreen_attn_ = !fullscreen_attn_;
        if (!fullscreen_attn_) exit_fullscreen();
    }
}

void TUI::handle_packet_key(int ch) {
    if (ch == 'j' || ch == KEY_DOWN) {
        stream_follow_ = false;
        stream_scroll_++;
    }
    if (ch == 'k' || ch == KEY_UP) {
        stream_follow_ = false;
        stream_scroll_ = std::max(0, stream_scroll_ - 1);
    }
    if (ch == 'G') stream_follow_ = true;
    if (ch == ' ') stream_follow_ = !stream_follow_;
}

void TUI::handle_key(int ch) {
    if (fullscreen_attn_) {
        handle_attention_key(ch);
        if (ch == 'f' || ch == 'F' || ch == 'q' || ch == 27) {
            exit_fullscreen();
        }
        return;
    }

    // Global.
    if (ch == 'q' || ch == 'Q') { running_ = false; return; }

    if (ch == '\t') {
        // Cycle focus.
        int next = (static_cast<int>(focus_) + 1) % static_cast<int>(Panel::COUNT);
        focus_ = static_cast<Panel>(next);
        return;
    }

    // Delegate to focused panel.
    switch (focus_) {
        case Panel::Topology:     handle_topology_key(ch); break;
        case Panel::PacketStream: handle_packet_key(ch);   break;
        case Panel::Attention:    handle_attention_key(ch); break;
        default: break;
    }
}

// ─── Main loop ────────────────────────────────────────────────────────────────

void TUI::run() {
    while (running_) {
        // Check terminal resize.
        int nr, nc;
        getmaxyx(stdscr, nr, nc);
        if (nr != rows_ || nc != cols_) {
            recalc_layout();
        }

        draw_all();

        // Non-blocking input — poll every 80 ms.
        int ch = getch();
        if (ch != ERR) handle_key(ch);

        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
}
