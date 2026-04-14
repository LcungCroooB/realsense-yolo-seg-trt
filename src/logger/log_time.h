#pragma once
#include <chrono>
#include <string>

namespace logging
{
    struct LogTimeConfig
    {
        int time_zone_hours = 8; // +8 默认
        bool use_utc = false;    // true 则忽略 time_zone_hours，使用 UTC
        std::string pattern = "<year>-<mon>-<day> <hour>:<min>:<sec>.<mili>";
    };

    struct DateParts
    {
        int year, month, day;
    };

    struct TimeParts
    {
        int hour, minute, second, millisecond;
    };

    class LogTimeFormatter final
    {
    private:
        LogTimeConfig config_;
    
    public:
        explicit LogTimeFormatter(LogTimeConfig config);
        void split(std::chrono::system_clock::time_point tp, DateParts &date, TimeParts &time) const;
        void format(std::chrono::system_clock::time_point tp, std::string &out) const;

        int dateKey(std::chrono::system_clock::time_point tp) const;
        std::string dateString(std::chrono::system_clock::time_point tp) const;
    };
}