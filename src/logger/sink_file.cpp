#include "sink.h"

#include <fstream>
#include <chrono>
#include <cerrno>
#include <mutex>
#include <sys/stat.h>

namespace logging
{
    namespace
    {
        static bool ensure_directory_exists(const std::string &dir)
        {
            // 目前支持linux
            if (dir.empty())
                return false;

            std::string cur;
            cur.reserve(dir.size());
            for (size_t i = 0; i < dir.size(); ++i)
            {
                const char c = dir[i];
                cur.push_back(c);
                if (c != '/')
                    continue;

                // 仅在路径分隔处创建目录，避免把 "./logs" 误拆成 "./l"、"./lo"...
                if (!cur.empty() && cur != "/" && cur != "./")
                {
                    if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
                        return false;
                }
            }

            if (!cur.empty() && cur.back() == '/')
                cur.pop_back();

            if (cur.empty())
                return false;

            if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
                return false;
            return true;
        }

        static std::string join_path(const std::string &dir, const std::string &filename)
        {
            if (dir.empty())
                return filename;
            if (dir.back() == '/')
                return dir + filename;
            return dir + "/" + filename;
        }

        class FileSink final : public iSink
        {
        private:
            FileSinkConfig config_;
            LogTimeFormatter time_formatter_;
            LogTimeFormatter rotate_time_;
            std::unique_ptr<iFormatter> formatter_;

            std::ofstream out_;
            long long current_key_ = -1; // 当前文件对应的日期键，格式为 YYYYMMDD
            std::string current_tag_;

            std::mutex mutex_;
            std::string scratch_;

            long long rotatekey(std::chrono::system_clock::time_point tp) const
            {
                const int mins = (config_.rotate_every_minutes > 0) ? config_.rotate_every_minutes : 1440;
                if (mins >= 1440)
                    return time_formatter_.dateKey(tp);

                DateParts date{};
                TimeParts time{};
                time_formatter_.split(tp, date, time);
                const long long minute_serial = static_cast<long long>(time_formatter_.dateKey(tp)) * 1440LL +
                                                static_cast<long long>(time.hour) * 60LL +
                                                static_cast<long long>(time.minute);
                return minute_serial / mins;
            }

            std::string rotateTag(std::chrono::system_clock::time_point tp) const
            {
                const int mins = (config_.rotate_every_minutes > 0) ? config_.rotate_every_minutes : 1440;
                if (mins >= 1440)
                    return time_formatter_.dateString(tp);

                std::string tag;
                tag.reserve(16);
                rotate_time_.format(tp, tag);
                return tag;
            }

            void rotateLocked(long long key, const std::string &tag)
            {
                if (out_.is_open())
                {
                    out_.flush();
                    out_.close();
                }

                current_key_ = key;
                current_tag_ = tag;

                const std::string filename = config_.base_filename + "_" + current_tag_ + ".log";
                const std::string path = join_path(config_.log_dir, filename);
                out_.open(path.c_str(), std::ios::out | std::ios::app);
            }

        public:
            FileSink(FileSinkConfig config, LogTimeConfig timeConfig)
                : config_(std::move(config)),
                  time_formatter_(timeConfig),
                  rotate_time_([&timeConfig]()
                               {
                        LogTimeConfig c = timeConfig;
                        c.pattern = "<year><mon><day><hour><min>"; // 旋转标签只精确到分钟
                        return c; }()),
                  formatter_(makePatternFormatter(std::move(timeConfig)))
            {
                ensure_directory_exists(config_.log_dir);
                scratch_.reserve(256);
            }

            void write(const LogEvent &event) override
            {
                const long long key = rotatekey(event.ts);
                const std::string tag = rotateTag(event.ts);

                std::lock_guard<std::mutex> lock(mutex_);
                if(!out_.is_open() || key != current_key_)
                    rotateLocked(key, tag);

                scratch_.clear();
                formatter_->formatInto(event, scratch_);
                out_ << scratch_ << "\n";

                if(config_.flush_on_every_write)
                    out_.flush();
                else if(config_.flush_on_error && (event.level == LogLevel::kError || event.level == LogLevel::kFatal))
                    out_.flush();
            }

            void flush() override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (out_.is_open())
                    out_.flush();
            }
        };
    }

    std::unique_ptr<iSink> makeFileSink(FileSinkConfig config, LogTimeConfig timeConfig)
    {
        return std::unique_ptr<iSink>(new FileSink(std::move(config), std::move(timeConfig)));
    }
}