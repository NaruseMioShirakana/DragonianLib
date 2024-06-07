#include "Util/ThreadPool.h"
#include "Util/Logger.h"

#ifdef _WIN32
#include <Windows.h>
#endif

LibSvcBegin
ThreadPool::~ThreadPool() {
    Stoped_ = true;
	Condition_.release((ptrdiff_t)Threads_.size());
    for (auto& CurTask : Threads_)
        if (CurTask.joinable()) CurTask.join();
}

void ThreadPool::Init(int64 _ThreadCount) {
#ifdef _WIN32
    if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
        SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
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
    {
	    Threads_.emplace_back(&ThreadPool::Run, this);
#ifdef _WIN32
        Threads_.back().get_id()._Get_underlying_id();
#endif
    }

    ThreadCount_ = _ThreadCount;
}

void ThreadPool::Run() {
#ifdef _WIN32
    LARGE_INTEGER Time1, Time2, Freq;
    QueryPerformanceFrequency(&Freq);
#endif
    while (!Stoped_) {
        Task task;
        {
            Condition_.acquire();
            std::unique_lock lock(Mx_);
            if (Tasks_.empty())
                continue;
            task = std::move(Tasks_.front());
            ++TaskProcessing_;
            Tasks_.pop();
        }
#ifdef WIN32
        if(LogTime_)
            QueryPerformanceCounter(&Time1);
#endif
        task();
#ifdef WIN32
        if (LogTime_)
        {
	        QueryPerformanceCounter(&Time2);
            LogMessage(("[Thread] Task Cost Time:" + std::to_string(double(Time2.QuadPart - Time1.QuadPart) * 1000. / (double)Freq.QuadPart) + "ms").c_str());
        }
#endif
        --TaskProcessing_;
        if (Tasks_.empty() && !TaskProcessing_ && Joinable)
            JoinCondition_.release();
    }
}

void ThreadPool::Join()
{
    std::lock_guard lg(JoinMx_);
    //Condition_.release();
    Joinable = true;
    JoinCondition_.acquire();
    Joinable = false;
    if (LogTime_)
        LogMessage("[Thread] All Task Finished!");
}

LibSvcEnd