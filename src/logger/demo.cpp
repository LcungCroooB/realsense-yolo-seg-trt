#include "logger_manager.h"
#include "logger_macro.h"

#include <chrono>
#include <thread>

int main()
{
    logging::LogManagerConfig config;
    config.global_lv = logging::LogLevel::kDebug;

    config.enable_console = true;
    config.console_config.to_stderr = false;

    config.enable_file = true;
    config.file_config.log_dir = "./logs";
    config.file_config.base_filename = "demo";
    config.file_config.rotate_every_minutes = 1440;

    config.async_config.queue_capacity = 1024;
    config.async_config.batch_size = 64;
    config.async_config.sync_on_error = true;

    logging::LoggerManager::instance().init(config);
    LOG_I("trt", "detector ready, score=%.2f", 0.93);
    LOG_D("tracking", "tracking frame=%d", 1001);
    LOG_W("planner", "path blocked, replan needed");
    LOG_E("control", "actuator failure: %s", "steering");

    std::this_thread::sleep_for(std::chrono::seconds(1)); // 等待异步日志线程处理完日志
    logging::LoggerManager::instance().shutdown();

    return 0;
}