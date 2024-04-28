#pragma once
#include "Base.h"
#include <thread>
#include <queue>
#include <future>
#include <mutex>
#include <semaphore>

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

        std::lock_guard lg(JoinMx_);

        auto task = std::make_shared<std::packaged_task<RetType()>>(
            std::bind(std::forward<_FunTy>(_Function), std::forward<_ArgsTy>(_Args)...));

        auto ret = task->get_future();
        {
            std::lock_guard lock(Mx_);
            Tasks_.emplace([task] { (*task)(); });
        }

        Condition_.release();

        return ret;
    }

    void Init(int64 _ThreadCount = 0);

    void Join();

    int64 GetThreadCount() const
    {
        return ThreadCount_;
    }

private:
    std::vector<std::thread> Threads_;
    std::atomic<bool> Stoped_;
    std::atomic<size_t> TaskProcessing_ = 0;
    std::mutex Mx_, JoinMx_;
    std::queue<Task> Tasks_;
    std::counting_semaphore<256> Condition_{ 0 }, JoinCondition_{ 0 };
    int64 ThreadCount_ = 0; 

    void Run();
};

LibSvcEnd