#include "logger_manager.h"

namespace logging
{
    LoggerManager &LoggerManager::instance()
    {
        static LoggerManager inst;
        return inst;
    }

    void LoggerManager::init(LogManagerConfig config)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (inited_)
            return;

        cfg_ = std::move(config);

        std::vector<std::unique_ptr<iSink>> sinks;
        if (cfg_.enable_console)
            sinks.push_back(makeConsoleSink(cfg_.console_config, cfg_.time_config));
        if (cfg_.enable_file)
            sinks.push_back(makeFileSink(cfg_.file_config, cfg_.time_config));

        backend_.reset(new AsyncBackend(cfg_.async_config, std::move(sinks)));
        backend_->start();

        inited_ = true;
    }

    void LoggerManager::shutdown()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!inited_)
            return;
        
        if (backend_)
            backend_->stop();
        backend_.reset();
        loggers_.clear();
        inited_ = false;
    }

    Logger *LoggerManager::getLogger(const std::string &category)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!inited_)
        {
            lock.unlock();
            init(LogManagerConfig{}); // 使用默认配置初始化
            lock.lock();
        }

        auto it = loggers_.find(category);
        if (it != loggers_.end())
            return it->second.get();

        auto logger = std::make_unique<Logger>(category, backend_.get());
        logger->setLevel(cfg_.global_lv);
        Logger *ptr = logger.get();
        loggers_.emplace(category, std::move(logger));
        return ptr;
    }

    void LoggerManager::setGlobalLevel(LogLevel level)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg_.global_lv = level;
        for (auto &kv : loggers_)
            kv.second->setLevel(level);
    }
}