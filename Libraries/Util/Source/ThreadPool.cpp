#include "Libraries/Util/ThreadPool.h"
#include "Libraries/Util/Logger.h"

#ifdef _WIN32
#include <Windows.h>
#endif

_D_Dragonian_Lib_Space_Begin

class ThreadLogger : public Logger
{
public:
    ThreadLogger() : Logger(
        *_D_Dragonian_Lib_Namespace GetDefaultLogger(),
        L"Thread-" + std::to_wstring(std::this_thread::get_id()._Get_underlying_id())
    )
    {

    }
};

ThreadPool::ThreadPool(
    Int64 _ThreadCount,
    std::wstring _Name,
    std::wstring _Desc
) : Stoped_(true), ThreadCount_(_ThreadCount), Name_(std::move(_Name)), Desc_(std::move(_Desc))
{
    Init(_ThreadCount);
}

ThreadPool::~ThreadPool()
{
    Join();
}

void ThreadPool::Init(Int64 _ThreadCount)
{
#ifdef _WIN32
    if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
        SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
    if (_ThreadCount == 0) _ThreadCount = ThreadCount_;
    if (_ThreadCount == 0) _ThreadCount = std::thread::hardware_concurrency();
    if (!Stoped_) Join();
    Stoped_ = false;
    Threads_.clear();
    for (Int64 i = 0; i < _ThreadCount; i++)
	    Threads_.emplace_back(&ThreadPool::Run, this);
    ThreadCount_ = _ThreadCount;
}

void ThreadPool::Run()
{
    auto Logger = ThreadLogger();
    auto Start = std::chrono::high_resolution_clock::now();
    const auto ThreadDescription =
        Name_ + L" {" + Desc_ +
        L", Id: " + std::to_wstring(std::this_thread::get_id()._Get_underlying_id()) + L"}";
#if _WIN32
    SetThreadDescription(GetCurrentThread(), ThreadDescription.c_str());
    if (GetThreadPriority(GetCurrentThread()) != THREAD_PRIORITY_TIME_CRITICAL)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
    while (!Stoped_ || !Tasks_.empty()) {
        Task task;
        {
            Condition_.acquire();
            std::lock_guard lock(TaskMx_);
            if (Tasks_.empty()) continue;
            task = std::move(Tasks_.front());
            Tasks_.pop();
        }
        if (LogTime_) Start = std::chrono::high_resolution_clock::now();
        task();
        if (LogTime_)
        {
            std::chrono::duration<double, std::milli> CostTime = std::chrono::high_resolution_clock::now() - Start;
            Logger.LogInfo(L"Task Cost Time:" + std::to_wstring(CostTime.count()) + L"ms");
        }
    }
}

void ThreadPool::Join()
{
    Stoped_ = true;
    Condition_.release((ptrdiff_t)std::max(Tasks_.size(), Threads_.size()));
    for (auto& CurTask : Threads_) if (CurTask.joinable()) CurTask.join();
    while (Condition_.try_acquire()) {}
    if (LogTime_) GetDefaultLogger()->LogInfo(L"All Task Finished!");
}

_D_Dragonian_Lib_Space_End