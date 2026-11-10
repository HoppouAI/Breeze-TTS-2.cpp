#include "breeze/model.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace breeze {

bool BreezeModel::load(const std::string & path, bool prefer_gpu, bool want_te_cpu) {
    backend.init(prefer_gpu);
    const char * te = std::getenv("BREEZE_TE_CPU");
    if (te && *te && strcmp(te, "0") != 0) want_te_cpu = true;
    te_cpu = backend.is_gpu && want_te_cpu;
    const char * embedding = std::getenv("BREEZE_TE_EMBED_CPU");
    te_embedding_cpu = backend.is_gpu && !te_cpu && embedding && *embedding && strcmp(embedding, "0") != 0;
    if (te_cpu || te_embedding_cpu) {
        cpu.init(false);
        const char * nt = std::getenv("BREEZE_TE_THREADS");
        int n = te_cpu ? (nt ? std::atoi(nt) : (int) std::max(1u, std::thread::hardware_concurrency() / 2)) : 1;
        ggml_backend_cpu_set_n_threads(cpu.backend, std::max(1, n));
        if (te_cpu) printf("text encoder: cpu, %d threads\n", std::max(1, n));
        else printf("text embedding lookup: cpu, 1 thread; text encoder: gpu\n");
    }
    const char * cpu_prefix = te_cpu ? "te." : te_embedding_cpu ? "te.token_embd.weight" : nullptr;
    if (!gg.load(path, backend, te_cpu || te_embedding_cpu ? &cpu : nullptr, cpu_prefix)) return false;
    cfg = parse_config(gg);
    if (!tok.load(gg)) return false;
    return true;
}

void BreezeModel::free() {
    gg.free();
    if (cpu.backend) cpu.free();
    backend.free();
}

}
