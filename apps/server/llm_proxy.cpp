#include "llm_proxy.h"

#include <cstdio>

namespace breeze {

// "http://host:port/v1" into the part httplib connects to and the path prefix in front of every call
static void split_url(const std::string & url, std::string & origin, std::string & prefix) {
    const size_t scheme = url.find("://");
    const size_t slash = url.find('/', scheme == std::string::npos ? 0 : scheme + 3);
    origin = slash == std::string::npos ? url : url.substr(0, slash);
    prefix = slash == std::string::npos ? "" : url.substr(slash);
    while (!prefix.empty() && prefix.back() == '/') prefix.pop_back();
}

void add_llm_proxy(httplib::Server & svr, const std::string & url) {
    std::string origin, prefix;
    split_url(url, origin, prefix);

    // the browser cant call most local llm servers itself without cors, so it goes through here
    const auto forward = [origin, prefix](const httplib::Request & req, httplib::Response & res) {
        const std::string path = prefix + "/" + req.matches[1].str();
        httplib::Headers headers;
        if (req.has_header("Authorization")) headers.emplace("Authorization", req.get_header_value("Authorization"));
        const std::string method = req.method;
        const std::string body = req.body;
        const std::string type = req.has_header("Content-Type") ? req.get_header_value("Content-Type") : "application/json";

        // a reply streams token by token, so pass every chunk on the moment it lands
        res.set_chunked_content_provider("text/event-stream", [=](size_t, httplib::DataSink & sink) {
            httplib::Client cli(origin);
            cli.set_connection_timeout(5);
            cli.set_read_timeout(300);
            httplib::Request up;
            up.method = method;
            up.path = path;
            up.headers = headers;
            up.body = body;
            if (!body.empty()) up.set_header("Content-Type", type);
            up.content_receiver = [&](const char * d, size_t n, uint64_t, uint64_t) { return sink.write(d, n); };
            const auto r = cli.send(up);
            if (!r) {
                const std::string msg = "{\"error\":{\"message\":\"could not reach the llm at " + origin + "\"}}";
                sink.write(msg.data(), msg.size());
            }
            sink.done();
            return true;
        });
    };
    svr.Get(R"(/llm/(.*))", forward);
    svr.Post(R"(/llm/(.*))", forward);
    printf("llm proxy: /llm/ -> %s\n", url.c_str());
}

} // namespace breeze
