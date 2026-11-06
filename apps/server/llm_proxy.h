#pragma once

#include "httplib.h"

#include <string>

namespace breeze {

// forwards /llm/<path> to an openai compatible server at url, streaming the reply back
void add_llm_proxy(httplib::Server & svr, const std::string & url);

} // namespace breeze
