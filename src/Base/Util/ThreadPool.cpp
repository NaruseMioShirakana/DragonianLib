#include "Util/ThreadPool.h"
#include "Util/Logger.h"

LibSvcBegin
ThreadPool::~ThreadPool() {
    Stoped_ = true;
    Condition_.notify_all();
    for (auto& CurTask : Threads_)
        if (CurTask.joinable()) CurTask.join();
}

void ThreadPool::Init(int64 _ThreadCount) {
    if (_ThreadCount == 0)
        _ThreadCount = ThreadCount_;
    if (_ThreadCount == 0)
        _ThreadCount = std::thread::hardware_concurrency();

    if (!Stoped_)
        LibSvcThrow("Thread Pool Already Start!");

    std::unique_lock lock(Mx_);
    if (!Stoped_)
        LibSvcThrow("Thread Pool Already Start!");
    Stoped_ = false;
    Threads_.clear();
    for (int64 i = 0; i < _ThreadCount; i++)
        Threads_.emplace_back(&ThreadPool::Run, this);

    ThreadCount_ = _ThreadCount;
}

void ThreadPool::Run() {
    while (true) {
        Task task;
        {
            std::unique_lock lock(Mx_);
            Condition_.wait(lock, [this] { return Stoped_ || !(Tasks_.empty()); });
            if (Tasks_.empty())
                return;
            task = std::move(Tasks_.front());
            Tasks_.pop();
        }
        task();
    }
}

void ThreadPool::Join()
{
    Stoped_ = true;
    Condition_.notify_all();
    for (auto& CurTask : Threads_)
        if (CurTask.joinable()) CurTask.join();
    Init();
}

LibSvcEnd