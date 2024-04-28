#include "Util/ThreadPool.h"
#include "Util/Logger.h"

LibSvcBegin
ThreadPool::~ThreadPool() {
    Stoped_ = true;
	Condition_.release((ptrdiff_t)Threads_.size());
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
    while (!Stoped_) {
        Task task;
        {
            Condition_.acquire();
            std::unique_lock lock(Mx_);
            if (Tasks_.empty())
                continue;
            task = std::move(Tasks_.front());
            Tasks_.pop();
        }
        ++TaskProcessing_;
        task();
        --TaskProcessing_;
        if (Tasks_.empty() && !TaskProcessing_)
            JoinCondition_.release();
    }
}

void ThreadPool::Join()
{
    std::lock_guard lg(JoinMx_);
    //Condition_.release();
    JoinCondition_.acquire();
}

LibSvcEnd