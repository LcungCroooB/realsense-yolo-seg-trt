#pragma once

namespace app_logger
{
    class LoggerRuntimeGuard final
    {
    public:
        LoggerRuntimeGuard();
        ~LoggerRuntimeGuard();

        LoggerRuntimeGuard(const LoggerRuntimeGuard &) = delete;
        LoggerRuntimeGuard &operator=(const LoggerRuntimeGuard &) = delete;
    };
}
