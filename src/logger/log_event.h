#pragma once
#include "log_level.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace logging
{
    using Fields = std::unordered_map<std::string, std::string>;

    struct SourceLocation
    {
        const char *file = ""; // 文件名
        int line = 0;          // 行号
        const char *func = ""; // 函数名
    };
    // __FILE__、__LINE__、__func__ 是 C++ 预定义的宏，分别表示当前文件名、行号和函数名，生命周期时静态的，编译时就确定了，不会有性能问题

    struct LogEvent
    {
        std::chrono::system_clock::time_point ts; // 时间戳
        LogLevel level = LogLevel::kInfo;         // 日志级别
        std::string category;                     // 分类标签（tensorrt/vslam/cpu/vla/...，可扩展）
        std::thread::id tid;                      // 线程
        SourceLocation src;                       // 日志参数
        std::string message;                      // 格式化信息
        Fields fields;                            // 结构化字段
    };
}