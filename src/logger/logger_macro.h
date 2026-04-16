#pragma once
#include "logger_manager.h"

#define LOG_SRC ::logging::SourceLocation{__FILE__, __LINE__, __func__}

#define LOG_IMPL(modular, level, ...)                                      \
	do                                                                      \
	{                                                                       \
		auto logger__ = ::logging::LoggerManager::instance().getLogger(modular); \
		if (logger__)                                                       \
			logger__->log(LOG_SRC, level, __VA_ARGS__);                    \
	} while (0)

#define LOG_T(modular, ...) LOG_IMPL(modular, ::logging::LogLevel::kTrace, __VA_ARGS__)
#define LOG_D(modular, ...) LOG_IMPL(modular, ::logging::LogLevel::kDebug, __VA_ARGS__)
#define LOG_I(modular, ...) LOG_IMPL(modular, ::logging::LogLevel::kInfo, __VA_ARGS__)
#define LOG_W(modular, ...) LOG_IMPL(modular, ::logging::LogLevel::kWarn, __VA_ARGS__)
#define LOG_E(modular, ...) LOG_IMPL(modular, ::logging::LogLevel::kError, __VA_ARGS__)
#define LOG_F(modular, ...) LOG_IMPL(modular, ::logging::LogLevel::kFatal, __VA_ARGS__)