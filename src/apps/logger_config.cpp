#include "logger_config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "yaml_helpers.hpp"
#include "../common/utils.hpp"
#include "../logger/logger_manager.h"

namespace
{
    std::string to_lower_copy(const std::string &text)
    {
        std::string out = text;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c)
        {
            return static_cast<char>(std::tolower(c));
        });
        return out;
    }

    bool parse_log_level(const std::string &text, logging::LogLevel &level)
    {
        const std::string value = to_lower_copy(text);
        if (value == "fatal")
        {
            level = logging::LogLevel::kFatal;
            return true;
        }
        if (value == "error")
        {
            level = logging::LogLevel::kError;
            return true;
        }
        if (value == "warn" || value == "warning")
        {
            level = logging::LogLevel::kWarn;
            return true;
        }
        if (value == "info")
        {
            level = logging::LogLevel::kInfo;
            return true;
        }
        if (value == "debug")
        {
            level = logging::LogLevel::kDebug;
            return true;
        }
        if (value == "trace")
        {
            level = logging::LogLevel::kTrace;
            return true;
        }
        return false;
    }

    bool parse_overflow_policy(const std::string &text, logging::OverflowPolicy &policy)
    {
        const std::string value = to_lower_copy(text);
        if (value == "block")
        {
            policy = logging::OverflowPolicy::block;
            return true;
        }
        if (value == "drop_oldest")
        {
            policy = logging::OverflowPolicy::drop_oldest;
            return true;
        }
        if (value == "drop_newest")
        {
            policy = logging::OverflowPolicy::drop_newest;
            return true;
        }
        return false;
    }

    bool try_load_logger_config(const std::string &path, logging::LogManagerConfig &config,
                                bool &found, std::string &message)
    {
        found = false;
        message.clear();

        std::string resolved;
        std::string prefix;
        if (!utils::path::resolve_input_path(path, resolved, prefix))
            return true;

        found = true;

        YAML::Node root;
        try
        {
            root = YAML::LoadFile(resolved);
        }
        catch (const YAML::Exception &e)
        {
            message = e.what();
            return false;
        }

        const YAML::Node logger_node = root["logger"] ? root["logger"] : root;

        logging::LogLevel parsed_level = config.global_lv;
        const std::string level_text = yaml_node::read_str(logger_node, "global_level", "info");
        if (parse_log_level(level_text, parsed_level))
            config.global_lv = parsed_level;

        const YAML::Node time_node = logger_node["time"];
        config.time_config.use_utc = yaml_node::read_bool(time_node, "use_utc", config.time_config.use_utc);
        config.time_config.time_zone_hours = yaml_node::read_int(time_node, "time_zone_hours", config.time_config.time_zone_hours);
        config.time_config.pattern = yaml_node::read_str(time_node, "pattern", config.time_config.pattern);

        const YAML::Node console_node = logger_node["console"];
        config.enable_console = yaml_node::read_bool(console_node, "enable", config.enable_console);
        config.console_config.to_stderr = yaml_node::read_bool(console_node, "to_stderr", config.console_config.to_stderr);

        const YAML::Node file_node = logger_node["file"];
        config.enable_file = yaml_node::read_bool(file_node, "enable", config.enable_file);
        config.file_config.log_dir = yaml_node::read_str(file_node, "log_dir", config.file_config.log_dir);
        config.file_config.base_filename = yaml_node::read_str(file_node, "base_filename", config.file_config.base_filename);
        config.file_config.rotate_every_minutes = yaml_node::read_int(file_node, "rotate_every_minutes", config.file_config.rotate_every_minutes);
        config.file_config.flush_on_every_write = yaml_node::read_bool(file_node, "flush_on_every_write", config.file_config.flush_on_every_write);
        config.file_config.flush_on_error = yaml_node::read_bool(file_node, "flush_on_error", config.file_config.flush_on_error);

        const YAML::Node async_node = logger_node["async"];
        const int queue_capacity = yaml_node::read_int(async_node, "queue_capacity", static_cast<int>(config.async_config.queue_capacity));
        const int batch_size = yaml_node::read_int(async_node, "batch_size", static_cast<int>(config.async_config.batch_size));
        config.async_config.queue_capacity = static_cast<std::size_t>(queue_capacity > 0 ? queue_capacity : 1);
        config.async_config.batch_size = static_cast<std::size_t>(batch_size > 0 ? batch_size : 1);
        config.async_config.sync_on_error = yaml_node::read_bool(async_node, "sync_on_error", config.async_config.sync_on_error);

        const std::string overflow_policy_text = yaml_node::read_str(async_node, "overflow_policy", "drop_oldest");
        logging::OverflowPolicy overflow_policy = config.async_config.overflow_policy;
        if (parse_overflow_policy(overflow_policy_text, overflow_policy))
            config.async_config.overflow_policy = overflow_policy;

        message = resolved;
        return true;
    }
}

namespace app_logger
{
    LoggerRuntimeGuard::LoggerRuntimeGuard()
    {
        logging::LogManagerConfig config;
        config.global_lv = logging::LogLevel::kInfo;
        config.enable_console = true;
        config.enable_file = false;

        const char *env_path = std::getenv("APP_LOGGER_CONFIG");
        const std::string config_path = (env_path && env_path[0] != '\0')
            ? std::string(env_path)
            : std::string("configs/logger.yaml");

        bool found = false;
        std::string message;
        const bool ok = try_load_logger_config(config_path, config, found, message);
        if (!ok)
        {
            std::cerr << "[logger] parse config failed: " << config_path
                      << ", fallback to defaults, error=" << message << std::endl;
        }
        else if (found)
        {
            std::cerr << "[logger] loaded config: " << message << std::endl;
        }

        logging::LoggerManager::instance().init(config);
    }

    LoggerRuntimeGuard::~LoggerRuntimeGuard()
    {
        logging::LoggerManager::instance().shutdown();
    }
}
