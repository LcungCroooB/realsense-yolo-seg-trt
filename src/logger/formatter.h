#pragma once
#include "log_event.h"
#include "log_time.h"

#include <memory>
#include <string>

namespace logging
{
    class iFormatter
    {
    public:
        virtual ~iFormatter() = default;
        virtual void formatInto(const LogEvent &event, std::string &out) = 0;
    };

    // 2026-03-08 16:20:30.123 [vision] file.cpp:42 I | message k=v
    std::unique_ptr<iFormatter> makePatternFormatter(LogTimeConfig timeConfig);
}