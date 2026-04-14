#include "logger.h"

#include <cstdio>
#include <thread>

namespace logging
{
    Logger::Logger(std::string category, AsyncBackend *backend)
        : category_(std::move(category)), backend_(backend) {}

    void Logger::setDefaultFields(std::string key, std::string value)
    {
        default_[std::move(key)] = std::move(value);
    }

    std::string Logger::vformat(const char *fmt, va_list args)
    {
        if (!fmt)
            return {};

        char stack_buffer[1024];
        std::va_list ap1;
        va_copy(ap1, args);
        int n = std::vsnprintf(stack_buffer, sizeof(stack_buffer), fmt, ap1);
        va_end(ap1);
        if (n < static_cast<int>(sizeof(stack_buffer)))
            return std::string(stack_buffer, (std::size_t)n);
        
        std::string out;
        out.resize(static_cast<std::size_t>(n));

        std::va_list ap2;
        va_copy(ap2, args);
        std::vsnprintf(&out[0], out.size(), fmt, ap2);
        va_end(ap2);

        out.resize(static_cast<std::size_t>(n)); // 去掉末尾的 null terminator
        return out;
    }

    void Logger::log(SourceLocation src, LogLevel lv, const char *fmt, ...)
    {
        if ((int)lv > (int)level_)
            return;

        va_list args;
        va_start(args, fmt);
        const std::string message = vformat(fmt, args);
        va_end(args);

        LogEvent e;
        e.ts = std::chrono::system_clock::now();
        e.level = lv;
        e.category = category_;
        e.tid = std::this_thread::get_id();
        e.src = src;
        e.message = std::move(message);

        if(backend_)
            backend_->submit(std::move(e));
    }

    void Logger::logwithFields(SourceLocation src, LogLevel lv, Fields fields, const char *fmt, ...)
    {
        if ((int)lv > (int)level_)
            return;

        va_list args;
        va_start(args, fmt);
        const std::string message = vformat(fmt, args);
        va_end(args);

        // 合并默认字段和传入的字段，传入的字段优先级更高
        Fields merged_fields = default_;
        merged_fields.insert(fields.begin(), fields.end());

        LogEvent e;
        e.ts = std::chrono::system_clock::now();
        e.level = lv;
        e.category = category_;
        e.tid = std::this_thread::get_id();
        e.src = src;
        e.message = std::move(message);

        e.fields = default_;
        for(auto &kv : fields)
            e.fields[kv.first] = std::move(kv.second);

        if(backend_)
            backend_->submit(std::move(e));
    }
}