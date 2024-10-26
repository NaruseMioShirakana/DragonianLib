/**
 * FileName: ThreadPool.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "Base.h"
#include <thread>
#include <queue>
#include <future>
#include <mutex>
#include <semaphore>

_D_Dragonian_Lib_Space_Begin

class ThreadPool {
public:
    using Task = std::function<void()>;

    ThreadPool(int64 _ThreadCount = 0) : Stoped_(true), ThreadCount_(_ThreadCount) {}
    ~ThreadPool();

    template <typename _FunTy, typename... _ArgsTy>
    auto Commit(_FunTy&& _Function, _ArgsTy &&... _Args) {
        using RetType = decltype(_Function(_Args...));

        if (Stoped_)
            _D_Dragonian_Lib_Throw_Exception("Thread Pool Is Not Initialized!");

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

    bool Enabled() const
    {
        return !Stoped_;
    }

    int64 GetThreadCount() const
    {
        return ThreadCount_;
    }

    operator ThreadPool* ()
    {
        return this;
    }

    void EnableTimeLogger(bool _Enabled)
    {
        LogTime_ = _Enabled;
    }

private:
    std::vector<std::thread> Threads_;
    std::atomic<bool> Stoped_;
    std::atomic<size_t> TaskProcessing_ = 0;
    std::mutex Mx_, JoinMx_;
    std::queue<Task> Tasks_;
    std::counting_semaphore<256> Condition_{ 0 }, JoinCondition_{ 0 };
    bool Joinable = false;
    int64 ThreadCount_ = 0;
    bool LogTime_ = false;

    void Run();
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&&) = delete;
	ThreadPool& operator=(ThreadPool&&) = delete;
};

_D_Dragonian_Lib_Space_End