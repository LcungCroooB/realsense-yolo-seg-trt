#include "async_backend.h"

namespace logging
{
    AsyncBackend::AsyncBackend(AsyncConfig config, std::vector<std::unique_ptr<iSink>> sinks)
        : config_(std::move(config)), sinks_(std::move(sinks))
    {
        if(config.queue_capacity == 0)
            config_.queue_capacity = 1;
        if(config.batch_size == 0)
            config_.batch_size = 1;
    }

    AsyncBackend::~AsyncBackend()
    {
        stop();
    }

    void AsyncBackend::start()
    {
        // CAS 确保只启动一次，避免重复起线程
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true))
            return;
        
        stop_ = false;
        worker_thread_ = std::thread(&AsyncBackend::workerLoop, this);
    }

    void AsyncBackend::stop()
    {
        if (!running_.load())
            return;

        stop_.store(true);
        cv_.notify_all();
        if (worker_thread_.joinable())
            worker_thread_.join();
        running_.store(false);
        flush();
    }

    void AsyncBackend::submit(LogEvent e)
    {
        // 关键日志绕过队列，立即写盘（可靠性优先）
        if(config_.sync_on_error && (e.level == LogLevel::kError || e.level == LogLevel::kFatal))
        {
            for (auto &sink : sinks_)
                sink->write(e);
            for (auto &sink : sinks_)
                sink->flush();
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (config_.overflow_policy == OverflowPolicy::block)
        {
            // 队列满时阻塞，直到有空位或停止
            cv_.wait(lock, [&]()
                    { return stop_.load() || queue_.size() < config_.queue_capacity; });
            if (stop_.load())
                return;
            queue_.push_back(std::move(e));
        }
        else if (config_.overflow_policy == OverflowPolicy::drop_newest)
        {
            // 队列满时直接丢当前日志
            if(queue_.size() >= config_.queue_capacity)
                return;
            queue_.push_back(std::move(e));
        }
        else 
        {
            // 队列满时淘汰最旧日志，保证最新日志优先
            if (queue_.size() >= config_.queue_capacity)
                queue_.pop_front();
            queue_.push_back(std::move(e));
        }
        lock.unlock();
        cv_.notify_one();
    }

    void AsyncBackend::flush()
    {
       for (auto &sink : sinks_)
            sink->flush();
    }

    void AsyncBackend::workerLoop()
    {
        std::vector<LogEvent> batch;
        batch.reserve(config_.batch_size);

        while (!stop_.load())
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]()
                        { return stop_.load() || !queue_.empty(); });
                if (stop_.load() && queue_.empty())
                    break;

                batch.clear();
                while(!queue_.empty() && batch.size() < config_.batch_size)
                {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                }

                // 唤醒 block 策略下等待入队的生产者线程
                cv_.notify_all();
            }

            // 批量写出，减少锁频率
            for (const auto &event : batch)
            {
                for (auto &sink : sinks_)
                    sink->write(event);
            }
        }

        // stop 后把残留队列全部写完
        for(;;)
        {
            LogEvent e;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if(queue_.empty())
                    break;
                e = std::move(queue_.front());
                queue_.pop_front();
            }
            for (auto &sink : sinks_)
                sink->write(e);
        }
        flush();
    }
}