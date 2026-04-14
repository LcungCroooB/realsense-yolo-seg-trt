#include "log_time.h"
#include "date.h"

#include <charconv>

namespace logging
{
    namespace
    {
        static std::chrono::system_clock::time_point applyTimeZone(std::chrono::system_clock::time_point tp,
                                                                   const LogTimeConfig &config)
        {
            if (config.use_utc)
                return tp;                                          // 使用 UTC 时间，不进行时区转换
            return tp + std::chrono::hours(config.time_zone_hours); // 应用时区偏移
        }

        static void append2(std::string &out, int value)
        {
            out.push_back(char('0' + (value / 10) % 10));
            out.push_back(char('0' + value % 10));
        }

        static void append3(std::string &out, int value)
        {
            out.push_back(char('0' + (value / 100) % 10));
            out.push_back(char('0' + (value / 10) % 10));
            out.push_back(char('0' + value % 10));
        }

        // 追加整数到字符串，使用 std::to_chars 进行高效转换
        static void appendint(std::string &out, int value)
        {
            char buffer[24];
            auto res = std::to_chars(buffer, buffer + sizeof(buffer), value);
            if (res.ec == std::errc())
                out.append(buffer, res.ptr);
        }

    }

    LogTimeFormatter::LogTimeFormatter(LogTimeConfig config) : config_(std::move(config)) {}

    void LogTimeFormatter::split(std::chrono::system_clock::time_point tp, DateParts &date, TimeParts &time) const
    {
        tp = applyTimeZone(tp, config_); // 应用时区转换

        auto dp = date::floor<date::days>(tp);
        date::year_month_day ymd{dp};
        auto tm = date::make_time(std::chrono::duration_cast<std::chrono::milliseconds>(tp - dp));

        date.year = static_cast<int>(ymd.year());
        date.month = static_cast<unsigned>(ymd.month());
        date.day = static_cast<unsigned>(ymd.day());

        time.hour = static_cast<int>(tm.hours().count());
        time.minute = static_cast<int>(tm.minutes().count());
        time.second = static_cast<int>(tm.seconds().count());
        time.millisecond = static_cast<int>(tm.subseconds().count());
    }

    void LogTimeFormatter::format(std::chrono::system_clock::time_point tp, std::string &out) const
    {
        DateParts date{};
        TimeParts time{};
        split(tp, date, time);

        const std::string &pattern = config_.pattern;
        out.reserve(out.size() + pattern.size() + 8); // 预留足够空间，避免频繁 realloc
        for (size_t i = 0; i < pattern.size(); ++i)
        {
            if (pattern[i] != '<')
            {
                out.push_back(pattern[i]);
                continue;
            }

            const size_t end = pattern.find('>', i + 1);
            if (end == std::string::npos)
            {
                out.push_back(pattern[i]);
                continue; // 没有找到匹配的 '>'，将 '<' 视为普通字符
            }

            const std::string key = pattern.substr(i, end - i + 1);
            if (key == "<year>")
                appendint(out, date.year);
            else if (key == "<mon>")
                append2(out, date.month);
            else if (key == "<day>")
                append2(out, date.day);
            else if (key == "<hour>")
                append2(out, time.hour);
            else if (key == "<min>")
                append2(out, time.minute);
            else if (key == "<sec>")
                append2(out, time.second);
            else if (key == "<mili>")
                append3(out, time.millisecond);
            else
            {
                out.push_back('<');
                out += key;
                out.push_back('>');
            }
            i = end; // 跳过已处理的占位符
        }
    }

    int LogTimeFormatter::dateKey(std::chrono::system_clock::time_point tp) const
    {
        DateParts date{};
        TimeParts time{};
        split(tp, date, time);
        return date.year * 10000 + date.month * 100 + date.day; // 生成 YYYYMMDD 格式的整数
    }

    std::string LogTimeFormatter::dateString(std::chrono::system_clock::time_point tp) const
    {
        DateParts date{};
        TimeParts time{};
        split(tp, date, time);
        std::string out;
        appendint(out, date.year);
        out.push_back('-');
        append2(out, date.month);
        out.push_back('-');
        append2(out, date.day);
        return out; // 返回 YYYY-MM-DD 格式的字符串
    }

} // namespace logging
