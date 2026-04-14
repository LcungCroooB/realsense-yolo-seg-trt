#pragma once

#include "formatter.h"

#include <memory>
#include <string>

namespace logging
{
    class iSink
    {
    public:
        virtual ~iSink() = default;
        virtual void write(const LogEvent &event) = 0;
        virtual void flush() = 0;
    };

    struct consoleSinkConfig
    {
        bool to_stderr = false; // true 则输出到 stderr，否则输出到 stdout
    };

    struct FileSinkConfig
    {
        std::string log_dir = "./logs";    // 日志目录
        std::string base_filename = "app"; // 基础文件名，最终文件名
        int rotate_every_minutes = 60;     // 每隔多少分钟创建一个新文件
        bool flush_on_every_write = false; // 每次写入后立即 flush，适合调试阶段，生产环境建议设置为 false 以提升性能
        bool flush_on_error = true;        // 仅在日志级别为 error 或更高时 flush，适合生产环境
    };

    std::unique_ptr<iSink> makeConsoleSink(consoleSinkConfig config, LogTimeConfig timeconfig);
    std::unique_ptr<iSink> makeFileSink(FileSinkConfig config, LogTimeConfig timeConfig);
}