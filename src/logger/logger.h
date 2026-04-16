#pragma once

#include "async_backend.h"

#include <atomic>
#include <cstdarg>
#include <memory>
#include <string>

namespace logging
{
    class Logger final
    {
    public: 
        Logger(std::string category, std::weak_ptr<AsyncBackend> backend);

        void setLevel(LogLevel level) {level_.store(level, std::memory_order_relaxed);}
        LogLevel level() const {return level_.load(std::memory_order_relaxed);} 
        
        void setDefaultFields(std::string key, std::string value);

        // 常规日志
        void log(SourceLocation src, LogLevel lv, const char* fmt, ...);

        // 结构化日志，fields 中的键值对会被格式化器以特定方式输出，适合机器解析
        void logwithFields(SourceLocation src, LogLevel lv, Fields fields, const char* fmt, ...);
    
    private:
        static std::string vformat(const char* fmt, va_list args);

    private:
        std::string category_;
        std::weak_ptr<AsyncBackend> backend_;
        std::atomic<LogLevel> level_{LogLevel::kInfo};
        Fields default_;
    };
}