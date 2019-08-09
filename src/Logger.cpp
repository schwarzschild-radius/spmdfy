#include <spmdfy/Logger.hpp>

namespace spmdfy {

    std::shared_ptr<spdlog::logger> Logger::m_logger;

	void Logger::initLogger()
	{
		spdlog::set_pattern("%^[%l]%$: %v");
		m_logger = spdlog::stdout_color_mt("SPMDFY");
		m_logger->set_level(spdlog::level::trace);
	}

}