#pragma once

#include "async_backend.h"
#include "logger.h"
#include "sink.h"

#include <memory>
#include <mutex>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace logging
{
    struct LogManagerConfig
    {
        LogLevel global_lv = LogLevel::kInfo; // 全局默认日志级别，Logger 可以覆盖这个级别
        LogTimeConfig time_config;            // 时间格式配置，所有 Logger 共享
        bool enable_console = true;           // 是否启用控制台输出
        consoleSinkConfig console_config;     // 控制台输出配置
        bool enable_file = false;             // 是否启用文件输出
        FileSinkConfig file_config;           // 文件输出配置
        AsyncConfig async_config;             // 异步后端配置
    };

    class LoggerManager final
    {
    public:
        static LoggerManager &instance();

        void init(LogManagerConfig config);
        void shutdown();

        std::shared_ptr<Logger> getLogger(const std::string &category);
        void setGlobalLevel(LogLevel level);

    private:
        LoggerManager() = default;
    
    private:
        std::mutex mutex_;
        bool inited_ = false;
        LogManagerConfig cfg_{};

        std::shared_ptr<AsyncBackend> backend_;
        std::unordered_map<std::string, std::shared_ptr<Logger>> loggers_;
    };
}