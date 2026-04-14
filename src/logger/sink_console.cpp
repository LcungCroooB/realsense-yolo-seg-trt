#include "sink.h"

#include <iostream>
#include <mutex>

namespace logging
{
    namespace
    {
        class consoleSink final : public iSink
        {
        private:
            consoleSinkConfig config_;
            std::unique_ptr<iFormatter> formatter_;
            std::mutex mutex_;
            std::string scratch_;

        public:
            consoleSink(consoleSinkConfig config, LogTimeConfig timeConfig)
                : config_(std::move(config)),
                  formatter_(makePatternFormatter(std::move(timeConfig)))
            {
                scratch_.reserve(256);
            }

            void write(const LogEvent &event) override
            {
                scratch_.clear();
                formatter_->formatInto(event, scratch_);

                std::lock_guard<std::mutex> lock(mutex_); // 确保多线程环境下输出不混乱
                if (config_.to_stderr)
                    std::cerr << scratch_ << "\n";
                else
                    std::cout << scratch_ << "\n";
            }

            void flush() override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (config_.to_stderr)
                    std::cerr.flush();
                else
                    std::cout.flush();
            }
        };
    }
    
    std::unique_ptr<iSink> makeConsoleSink(consoleSinkConfig config, LogTimeConfig timeConfig)
    {
        return std::unique_ptr<iSink>(new consoleSink(std::move(config), std::move(timeConfig)));
    }
}
