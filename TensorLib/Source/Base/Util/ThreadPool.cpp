#include "Util/ThreadPool.h"
#include "Util/Logger.h"

#ifdef _WIN32
#include <Windows.h>
#endif

_D_Dragonian_Lib_Space_Begin

ThreadPool::ThreadPool(int64 _ThreadCount) : Stoped_(true), ThreadCount_(_ThreadCount) {
    Init(_ThreadCount);
}

ThreadPool::~ThreadPool() {
    Join();
}

void ThreadPool::Init(int64 _ThreadCount) {
#ifdef _WIN32
    if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
        SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
    if (_ThreadCount == 0) _ThreadCount = ThreadCount_;
    if (_ThreadCount == 0) _ThreadCount = std::thread::hardware_concurrency();
    if (!Stoped_) Join();
    Stoped_ = false;
    Threads_.clear();
    for (int64 i = 0; i < _ThreadCount; i++)
        Threads_.emplace_back(&ThreadPool::Run, this);
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
            std::lock_guard lock(TaskMx_);
            if (Tasks_.empty()) continue;
            task = std::move(Tasks_.front());
            Tasks_.pop();
        }
#ifdef WIN32
        if (GetThreadPriority(GetCurrentThread()) != THREAD_PRIORITY_TIME_CRITICAL)
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        if (LogTime_) QueryPerformanceCounter(&Time1);
#endif
        task();
#ifdef WIN32
        if (LogTime_)
        {
            QueryPerformanceCounter(&Time2);
            LogInfo(L"Task Cost Time:" + std::to_wstring(double(Time2.QuadPart - Time1.QuadPart) * 1000. / (double)Freq.QuadPart) + L"ms");
        }
#endif
    }
}

void ThreadPool::Join()
{
    Stoped_ = true;
    Condition_.release((ptrdiff_t)Threads_.size());
    for (auto& CurTask : Threads_) if (CurTask.joinable()) CurTask.join();
	while (Condition_.try_acquire()) {}
    if (LogTime_) LogInfo(L"All Task Finished!");
}

_D_Dragonian_Lib_Space_End