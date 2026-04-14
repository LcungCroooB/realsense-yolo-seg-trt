#pragma once
#include "logger_manager.h"

#define LOG_SRC ::logging::SourceLocation{__FILE__, __LINE__, __func__}

#define LOG_T(modular, ...) ::logging::LoggerManager::instance().getLogger(modular)->log(LOG_SRC, ::logging::LogLevel::kTrace, __VA_ARGS__)
#define LOG_D(modular, ...) ::logging::LoggerManager::instance().getLogger(modular)->log(LOG_SRC, ::logging::LogLevel::kDebug, __VA_ARGS__)
#define LOG_I(modular, ...) ::logging::LoggerManager::instance().getLogger(modular)->log(LOG_SRC, ::logging::LogLevel::kInfo, __VA_ARGS__)
#define LOG_W(modular, ...) ::logging::LoggerManager::instance().getLogger(modular)->log(LOG_SRC, ::logging::LogLevel::kWarn, __VA_ARGS__)
#define LOG_E(modular, ...) ::logging::LoggerManager::instance().getLogger(modular)->log(LOG_SRC, ::logging::LogLevel::kError,__VA_ARGS__)
#define LOG_F(modular, ...) ::logging::LoggerManager::instance().getLogger(modular)->log(LOG_SRC, ::logging::LogLevel::kFatal,__VA_ARGS__)