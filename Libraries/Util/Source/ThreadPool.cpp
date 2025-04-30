﻿#include "Libraries/Util/ThreadPool.h"
#include "Libraries/Util/Logger.h"

#ifdef _WIN32
#include <Windows.h>
#endif

_D_Dragonian_Lib_Space_Begin

class ThreadLogger : public Logger
{
	ThreadLogger(const ThreadLogger&) = delete;
	ThreadLogger(ThreadLogger&&) = delete;
	ThreadLogger& operator=(const ThreadLogger&) = delete;
	ThreadLogger& operator=(ThreadLogger&&) = delete;
    ThreadLogger() : Logger(*_D_Dragonian_Lib_Namespace GetDefaultLogger(), L"Thread")
    {
	    
    }

};

static DLogger& GetThreadLogger(Int64 ThreadId) noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
        *_D_Dragonian_Lib_Namespace GetDefaultLogger(),
		L"Thread: [" + std::to_wstring(ThreadId) + L"]"
    );
    return _MyLogger;
}

ThreadPool::ThreadPool(Int64 _ThreadCount) : Stoped_(true), ThreadCount_(_ThreadCount) {
    Init(_ThreadCount);
}

ThreadPool::~ThreadPool() {
    Join();
}

void ThreadPool::Init(Int64 _ThreadCount) {
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

void ThreadPool::Run() {
    auto Start = std::chrono::high_resolution_clock::now();
    while (!Stoped_ || !Tasks_.empty()) {
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
#endif
        if (LogTime_) Start = std::chrono::high_resolution_clock::now();
        task();
        if (LogTime_)
        {
            std::chrono::duration<double, std::milli> CostTime = std::chrono::high_resolution_clock::now() - Start;
            GetThreadLogger(std::this_thread::get_id()._Get_underlying_id())->LogInfo(L"Task Cost Time:" + std::to_wstring(CostTime.count()) + L"ms");
        }
    }
}

void ThreadPool::Join()
{
    Stoped_ = true;
    Condition_.release((ptrdiff_t)std::max(Tasks_.size(), Threads_.size()));
    for (auto& CurTask : Threads_) if (CurTask.joinable()) CurTask.join();
    while (Condition_.try_acquire()) {}
    if (LogTime_) GetThreadLogger(std::this_thread::get_id()._Get_underlying_id())->LogInfo(L"All Task Finished!");
}

_D_Dragonian_Lib_Space_End