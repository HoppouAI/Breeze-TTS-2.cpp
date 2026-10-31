#include "breeze/scheduler.h"

#include <algorithm>

namespace breeze {

void GenScheduler::start(BreezeModel & m, int parallel) {
    stop();
    m_model = &m;
    m_parallel = parallel < 1 ? 1 : parallel;
    m_slots = m_parallel;
    m_stop = false;
    m_thread = std::thread([this] { loop(); });
}

void GenScheduler::stop() {
    if (m_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(m_mu);
            m_stop = true;
        }
        m_cv.notify_all();
        m_thread.join();
    }
    m_pool.free();
}

bool GenScheduler::try_acquire(int n) {
    std::lock_guard<std::mutex> lock(m_slot_mu);
    if (m_slots < n) return false;
    m_slots -= n;
    return true;
}

void GenScheduler::acquire(int n) {
    std::unique_lock<std::mutex> lock(m_slot_mu);
    m_slot_cv.wait(lock, [&] { return m_slots >= n; });
    m_slots -= n;
}

void GenScheduler::release(int n) {
    {
        std::lock_guard<std::mutex> lock(m_slot_mu);
        m_slots += n;
    }
    m_slot_cv.notify_all();
}

void GenScheduler::exec(const std::function<void()> & fn) {
    std::mutex done_mu;
    std::condition_variable done_cv;
    bool done = false;
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(m_mu);
        m_tasks.push_back([&] {
            try { fn(); }
            catch (...) { error = std::current_exception(); }
            std::lock_guard<std::mutex> dl(done_mu);
            done = true;
            done_cv.notify_all();
        });
    }
    m_cv.notify_all();
    std::unique_lock<std::mutex> dl(done_mu);
    done_cv.wait(dl, [&] { return done; });
    if (error) std::rethrow_exception(error);
}

bool GenScheduler::run(SpeechJob & job) {
    Entry e;
    e.job = &job;
    const AudioCallback forward = job.cb;
    job.cb = [this, &e](const float * s, int n) {
        std::lock_guard<std::mutex> lock(m_mu);
        e.audio.emplace_back(s, s + n);
        e.cv.notify_all();
        return !e.cancel.load();
    };
    e.queued = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_mu);
        m_incoming.push_back(&e);
    }
    m_cv.notify_all();

    std::unique_lock<std::mutex> lock(m_mu);
    for (;;) {
        e.cv.wait(lock, [&] { return e.done || !e.audio.empty(); });
        if (e.audio.empty()) break;
        std::vector<float> chunk = std::move(e.audio.front());
        e.audio.pop_front();
        lock.unlock();
        const bool keep = forward(chunk.data(), (int) chunk.size());
        lock.lock();
        if (!keep) e.cancel = true;
    }
    if (e.error) std::rethrow_exception(e.error);
    return !e.cancel.load() && !job.stopped;
}

void GenScheduler::finish(Entry & e, std::exception_ptr error) {
    try {
        if (!error) e.job->finish();
    } catch (...) {
        error = std::current_exception();
    }
    if (error) e.job->abandon();
    std::lock_guard<std::mutex> lock(m_mu);
    e.error = error;
    e.done = true;
    e.cv.notify_all();
}

int GenScheduler::rows_busy() const {
    int rows = 0;
    for (const Entry * e : m_active) rows += e->job->branches();
    for (const Entry * e : m_incoming) rows += e->job->req.cfg_scale != 1.0f ? 2 : 1;
    return rows;
}

// starting a request stalls every stream for its whole prompt, so one thats about to run dry gets a
// head start first. the wait is bounded so a stream that never catches up cant starve a new one
bool GenScheduler::hold_back(const Entry & fresh) const {
    const auto now = std::chrono::steady_clock::now();
    if (now - fresh.queued > std::chrono::milliseconds(400)) return false;
    for (const Entry * e : m_active) {
        const SpeechJob & j = *e->job;
        if (!j.req.pace || !j.wants_frame()) continue;
        if (!j.clock.started || j.clock.runs_out - now < std::chrono::milliseconds(250)) return true;
    }
    return false;
}

void GenScheduler::loop() {
    for (;;) {
        std::function<void()> task;
        Entry * fresh = nullptr;
        int other_rows = 0;
        {
            std::unique_lock<std::mutex> lock(m_mu);
            m_cv.wait(lock, [&] {
                return m_stop || !m_tasks.empty() || !m_incoming.empty() || !m_active.empty();
            });
            if (m_stop && m_tasks.empty() && m_incoming.empty() && m_active.empty()) return;
            if (!m_tasks.empty()) {
                task = std::move(m_tasks.front());
                m_tasks.pop_front();
            } else if (!m_incoming.empty() && !hold_back(*m_incoming.front())) {
                fresh = m_incoming.front();
                m_incoming.pop_front();
                other_rows = rows_busy();
            }
        }
        if (task) task();

        if (fresh) {
            try {
                fresh->job->start(other_rows);
                m_active.push_back(fresh);
            } catch (...) {
                finish(*fresh, std::current_exception());
            }
        }

        std::vector<SpeechJob *> batch;
        for (Entry * e : m_active) {
            if (e->cancel) e->job->stopped = true;
            if (e->job->wants_frame()) batch.push_back(e->job);
        }
        std::exception_ptr failed;
        if (!batch.empty()) {
            try {
                step_jobs(*m_model, m_pool, batch);
            } catch (...) {
                failed = std::current_exception();
            }
        }
        for (auto it = m_active.begin(); it != m_active.end();) {
            Entry * e = *it;
            if (failed || !e->job->wants_frame()) {
                finish(*e, failed);
                it = m_active.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void GenScheduler::warm_up(const GenRequest & base) {
    exec([&] {
        GenRequest req = base;
        req.text = warm_up_text();
        req.instruction = GenRequest().instruction;
        req.max_new_tokens = 8;
        req.pace = false;
        const AudioCallback drop = [](const float *, int) { return true; };
        const auto batch_of = [&](int streams) {
            GenTimings tm;
            PlayClock clock;
            std::deque<SpeechJob> jobs;
            std::vector<SpeechJob *> all;
            for (int s = 0; s < streams; s++) {
                jobs.emplace_back(*m_model, m_pool, req, req.text, ChunkRef(), (uint32_t) s, drop, tm,
                                  std::chrono::steady_clock::now(), clock);
                jobs.back().start();
                all.push_back(&jobs.back());
            }
            for (;;) {
                std::vector<SpeechJob *> live;
                for (SpeechJob * j : all)
                    if (j->wants_frame()) live.push_back(j);
                if (live.empty()) break;
                step_jobs(*m_model, m_pool, live);
            }
            for (SpeechJob * j : all) j->finish();
        };
        req.cfg_scale = 1.0f;
        for (int streams = 1; streams <= m_parallel; streams++) batch_of(streams);
        req.cfg_scale = 2.0f;
        for (int streams = 1; streams <= std::max(1, m_parallel / 2); streams++) batch_of(streams);
    });
}

}
