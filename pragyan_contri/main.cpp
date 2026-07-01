#include "instrumentation.h"
#include "tui.h"
#include <iostream>
#include <string>
#include <vector>
#include <csignal>

// ─── Graceful shutdown on SIGINT ─────────────────────────────────────────────

static bool g_running = true;

void sig_handler(int) {
    g_running = false;
    InstrumentationEngine::instance().stop_simulation();
}

// ─── Usage ───────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cerr << "\nLLM Instrumentation, Tracing & Replay Platform\n"
              << "────────────────────────────────────────────────\n"
              << "Usage:\n"
              << "  " << prog << " [--model <name>] [--tokens <t1,t2,...>]\n\n"
              << "Flags:\n"
              << "  --model  <name>     Model name shown in header (default: llama-3-8b)\n"
              << "  --tokens <list>     Comma-separated token list for attention viz\n"
              << "                      (default: I,want,it,to,be,keyboard,driven)\n\n"
              << "Keyboard:\n"
              << "  Tab           Cycle focus between panels\n"
              << "  j/k           Navigate topology / scroll stream\n"
              << "  Space         Select layer (topology) / toggle follow (stream)\n"
              << "  h/l           Pan attention matrix left/right\n"
              << "  H/L           Prev/Next attention head\n"
              << "  +/-           Increase/decrease attention contrast\n"
              << "  F             Toggle fullscreen attention view\n"
              << "  Q             Quit\n\n";
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    // Parse args.
    std::string model_name = "llama-3-8b";
    std::vector<std::string> tokens = {
        "I", "want", "it", "to", "be", "keyboard", "driven"
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--model" && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            std::string tok_str = argv[++i];
            tokens.clear();
            std::string cur;
            for (char c : tok_str) {
                if (c == ',') {
                    if (!cur.empty()) tokens.push_back(cur);
                    cur.clear();
                } else {
                    cur += c;
                }
            }
            if (!cur.empty()) tokens.push_back(cur);
        }
    }

    if (tokens.empty()) {
        tokens = {"Hello", "world", "from", "LLM", "tracer"};
    }

    // Start simulation.
    auto& eng = InstrumentationEngine::instance();
    eng.start_simulation(model_name, tokens);

    // Run TUI.
    {
        TUI tui;
        tui.run();
    }

    // Stop simulation.
    eng.stop_simulation();
    return 0;
}
