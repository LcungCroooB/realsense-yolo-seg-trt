#pragma once
#include <cstdint>

namespace logging
{
    enum class LogLevel : std::uint8_t
    {
        kFatal = 0,
        kError = 1,
        kWarn = 2,
        kInfo = 3,
        kDebug = 4,
        kTrace = 5,
    };

    inline const char *ToString(LogLevel lv) noexcept  // 不抛出异常
    {
        switch (lv)
        {
        case LogLevel::kFatal:
            return "fatal";
        case LogLevel::kError:
            return "error";
        case LogLevel::kWarn:
            return "warn";
        case LogLevel::kInfo:
            return "info";
        case LogLevel::kDebug:
            return "debug";
        case LogLevel::kTrace:
            return "trace";
        default:
            return "unknown";
        }
    }
}