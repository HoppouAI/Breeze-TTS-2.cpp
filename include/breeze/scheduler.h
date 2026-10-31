#pragma once

#include "breeze/generation.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

namespace breeze {

// every frame of every stream runs on one thread, so streams can share a batch and nothing races the gpu
class GenScheduler {
public:
    ~GenScheduler() { stop(); }

    void start(BreezeModel & m, int parallel);
    void stop();
    int parallel() const { return m_parallel; }
    GenPool & pool() { return m_pool; }

    // a request holds slots for as long as it generates, try fails when too few are free
    bool try_acquire(int n = 1);
    void acquire(int n = 1);
    void release(int n = 1);

    // cfg runs every frame twice, so a stream with it costs what two without it do
    int cost(float cfg_scale) const { return cfg_scale != 1.0f && m_parallel > 1 ? 2 : 1; }

    // runs fn on the gpu thread between frames and waits for it
    void exec(const std::function<void()> & fn);

    // blocks until the job is done, the job's callback runs on the calling thread so a slow client
    // only holds up itself
    bool run(SpeechJob & job);

    // builds the batched graphs ahead of the first request that needs them
    void warm_up(const GenRequest & base);

private:
    struct Entry {
        SpeechJob * job = nullptr;
        std::deque<std::vector<float>> audio;
        bool done = false;
        std::exception_ptr error;
        std::atomic<bool> cancel{false};
        std::condition_variable cv;
        std::chrono::steady_clock::time_point queued;
    };

    void loop();
    int rows_busy() const;
    bool hold_back(const Entry & fresh) const;
    void finish(Entry & e, std::exception_ptr error);

    BreezeModel * m_model = nullptr;
    GenPool m_pool;
    int m_parallel = 1;

    std::mutex m_slot_mu;
    std::condition_variable m_slot_cv;
    int m_slots = 0;

    std::mutex m_mu;
    std::condition_variable m_cv;
    std::deque<std::function<void()>> m_tasks;
    std::deque<Entry *> m_incoming;
    std::list<Entry *> m_active;
    bool m_stop = false;
    std::thread m_thread;
};

}
