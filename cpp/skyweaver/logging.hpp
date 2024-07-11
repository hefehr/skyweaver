#ifndef SKYWEAVER_LOGGING_HPP
#define SKYWEAVER_LOGGING_HPP

#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/attributes/constant.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/attributes/timer.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <string>

namespace skyweaver
{

void init_logging(std::string const& severity)
{
    using namespace boost::log;

    add_common_attributes();
    core::get()->add_global_attribute("Scope", attributes::named_scope());

    add_console_log(
        std::cout,
        keywords::format =
            (expressions::stream
             << "[" << expressions::attr<boost::posix_time::ptime>("TimeStamp")
             << "] " << "["
             << expressions::attr<trivial::severity_level>("Severity") << "] "
             << "["
             << expressions::format_named_scope("Scope",
                                                keywords::format = "%n",
                                                keywords::depth  = 1)
             << "] " << expressions::smessage));

    if(severity == "debug") {
        core::get()->set_filter(trivial::severity >= trivial::debug);
    } else if(severity == "info") {
        core::get()->set_filter(trivial::severity >= trivial::info);
    } else if(severity == "warning") {
        core::get()->set_filter(trivial::severity >= trivial::warning);
    } else {
        core::get()->set_filter(trivial::severity >= trivial::error);
    }
}

} // namespace skyweaver

#endif // SKYWEAVER_LOGGING_HPP