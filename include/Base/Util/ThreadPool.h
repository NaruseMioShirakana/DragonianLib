#pragma once
#include "Base.h"
#include <thread>
#include <queue>
#include <future>
#include <mutex>

LibSvcBegin

class ThreadPool {
public:
    using Task = std::function<void()>;

    ThreadPool(int64 _ThreadCount = 0) : Stoped_(true), ThreadCount_(_ThreadCount) {}
    ~ThreadPool();

    template <typename _FunTy, typename... _ArgsTy>
    auto Commit(_FunTy&& _Function, _ArgsTy &&... _Args) {
        using RetType = decltype(_Function(_Args...));

        if (Stoped_)
            LibSvcThrow("Thread Pool Is Not Initialized!");

        auto task = std::make_shared<std::packaged_task<RetType()>>(
            std::bind(std::forward<_FunTy>(_Function), std::forward<_ArgsTy>(_Args)...));

        auto ret = task->get_future();
        {
            std::lock_guard lock(Mx_);
            Tasks_.emplace([task] { (*task)(); });
        }

        Condition_.notify_one();

        return ret;
    }

    void Init(int64 _ThreadCount = 0);

    void Join();

private:
    std::vector<std::thread> Threads_;
    std::atomic<bool> Stoped_;
    std::mutex Mx_;
    std::queue<Task> Tasks_;
    std::condition_variable Condition_;
    int64 ThreadCount_ = 0; 

    void Run();
};

LibSvcEnd