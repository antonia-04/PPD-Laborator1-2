#pragma once

#include <mutex>
#include <condition_variable>

class Barrier {
public:
    explicit Barrier(int count) : thread_count(count), counter(0), generation(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int current_gen = generation;
        counter++;
        if (counter == thread_count) {
            generation++;
            counter = 0;
            cv.notify_all();
        } else {
            // asteapta pana cand generatia se schimba
            cv.wait(lock, [this, current_gen] { return generation != current_gen; });
        }
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int thread_count;
    int counter;
    int generation;
};