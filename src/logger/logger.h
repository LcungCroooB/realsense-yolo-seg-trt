#pragma once

#include "async_backend.h"

#include <cstdarg>
#include <string>

namespace logging
{
    class Logger final
    {
    public: 
        Logger(std::string category, AsyncBackend *backend);

        void setLevel(LogLevel level) {level_ = level;}
        LogLevel level() const {return level_;}
        
        void setDefaultFields(std::string key, std::string value);

        // 常规日志
        void log(SourceLocation src, LogLevel lv, const char* fmt, ...);

        // 结构化日志，fields 中的键值对会被格式化器以特定方式输出，适合机器解析
        void logwithFields(SourceLocation src, LogLevel lv, Fields fields, const char* fmt, ...);
    
    private:
        static std::string vformat(const char* fmt, va_list args);

    private:
        std::string category_;
        AsyncBackend *backend_{nullptr};
        LogLevel level_{LogLevel::kInfo};
        Fields default_;
    };
}