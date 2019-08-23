#ifndef SPMDFY_LOGGER_HPP
#define SPMDFY_LOGGER_HPP

#include <memory>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace spmdfy {
/**
 * \class Logger
 * \ingroup Utility
 *
 * \brief Logger class for logging all things in the command line using spdlog
 * as fast logging frameworks which also uses a fmt string formatting library
 *
 * */
class Logger {
  public:

    /// initializes the logger
    static void initLogger();

    /// returns the logger singleton object
    static inline auto getSpmdfyLogger() -> std::shared_ptr<spdlog::logger> & {
        return m_logger;
    }

  private:
    static std::shared_ptr<spdlog::logger> m_logger;
};
} // namespace spmdfy

/**
 * \ingroup Utility
 *
 * \brief Logger macros to wrap around spdlog's logging functions
 *
 * */
// clang-format off
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
// clang-format on
#endif