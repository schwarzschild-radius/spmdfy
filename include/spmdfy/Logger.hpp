#ifndef SPMDFY_LOGGER_HPP
#define SPMDFY_LOGGER_HPP

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>

namespace spmdfy {

	class Logger
	{
	public:
		static void initLogger();
		static inline auto getSpmdfyLogger() -> std::shared_ptr<spdlog::logger>& { return m_logger; }
	private:
		static std::shared_ptr<spdlog::logger> m_logger;
	};
}

#ifdef SPMDFY_DEBUG
#define SPMDFY_TRACE(...)    ::spmdfy::Logger::getSpmdfyLogger()->trace(__VA_ARGS__)
#define SPMDFY_INFO(...)     ::spmdfy::Logger::getSpmdfyLogger()->info(__VA_ARGS__)
#define SPMDFY_WARN(...)     ::spmdfy::Logger::getSpmdfyLogger()->warn(__VA_ARGS__)
#define SPMDFY_ERROR(...)    ::spmdfy::Logger::getSpmdfyLogger()->error(__VA_ARGS__)
#define SPMDFY_CRITICAL(...) ::spmdfy::Logger::getSpmdfyLogger()->critical(__VA_ARGS__)
#else
#define SPMDFY_TRACE(...)
#define SPMDFY_INFO(...)
#define SPMDFY_WARN(...)
#define SPMDFY_ERROR(...)
#define SPMDFY_CRITICAL(...)
#endif

#endif