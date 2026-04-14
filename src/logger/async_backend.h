#pragma once

#include "sink.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace logging
{
    enum class OverflowPolicy
    {
        block,       // 队列满时阻塞写入线程，直到有空间
        drop_oldest, // 队列满时丢弃最旧的日志
        drop_newest, // 队列满时丢弃最新的日志
    };

    struct AsyncConfig
    {
        std::size_t queue_capacity = 4096;                            // 队列容量，超过后根据 overflow_policy 处理
        std::size_t batch_size = 128;                                 // 每次从队列中取出的日志数量，过大可能增加延迟，过小可能降低吞吐量
        OverflowPolicy overflow_policy = OverflowPolicy::drop_oldest; // 队列满时的处理策略
        bool sync_on_error = true;                                    // 当日志级别为 error 或更高时，强制同步写入以确保日志不丢失
    };

    class AsyncBackend final
    {
    public:
        AsyncBackend(AsyncConfig config, std::vector<std::unique_ptr<iSink>> sinks);
        ~AsyncBackend();

        AsyncBackend(const AsyncBackend &) = delete;
        AsyncBackend &operator=(const AsyncBackend &) = delete;

        void start();
        void stop();

        void submit(LogEvent e);
        void flush();

    private:
        void workerLoop(); // 后台线程函数，负责从队列中取出日志并写入到sinks中

    private:
        AsyncConfig config_;
        std::vector<std::unique_ptr<iSink>> sinks_;

        std::deque<LogEvent> queue_;
        std::mutex mutex_;
        std::condition_variable cv_;

        std::thread worker_thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> stop_{false};
    };
}