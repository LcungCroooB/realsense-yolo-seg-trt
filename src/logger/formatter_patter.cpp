#include "formatter.h"
#include "log_level.h"

namespace logging
{
    namespace
    {
        static char levelChar(LogLevel lv) noexcept
        {
            switch (lv)
            {
            case LogLevel::kTrace:
                return 'T';
            case LogLevel::kDebug:
                return 'D';
            case LogLevel::kInfo:
                return 'I';
            case LogLevel::kWarn:
                return 'W';
            case LogLevel::kError:
                return 'E';
            case LogLevel::kFatal:
                return 'F';
            default:
                return '?';
            }
        }

        static std::string basefilename(const char *path)
        {
            if (!path)
                return {};
            std::string s(path);
            const auto pos = s.find_last_of("/\\");
            return (pos == std::string::npos) ? s : s.substr(pos + 1);
        }

        class patternFormatter final : public iFormatter
        {
        private:
            LogTimeFormatter timeFormatter_;

        public:
            explicit patternFormatter(LogTimeConfig timeConfig) : timeFormatter_(std::move(timeConfig)) {}

            void formatInto(const LogEvent &event, std::string &out) override
            {
                out.reserve(out.size() + 128);
                timeFormatter_.format(event.ts, out);
                out.push_back(' ');
                out.push_back('[');
                out += event.category;
                out.push_back(']');
                out.push_back(' ');
                out += basefilename(event.src.file);
                out.push_back(':');
                out += std::to_string(event.src.line);
                out.push_back(' ');
                out.push_back(levelChar(event.level));
                out += " | ";
                out += event.message;

                if (!event.fields.empty())
                {
                    out += " | ";
                    for (const auto &kv : event.fields)
                    {
                        out.push_back(' ');
                        out += kv.first;
                        out.push_back('=');
                        out += kv.second;
                    }
                }
            }
        };
    }

    std::unique_ptr<iFormatter> makePatternFormatter(LogTimeConfig timeConfig)
    {
        return std::unique_ptr<iFormatter>(new patternFormatter(timeConfig));
    }

}