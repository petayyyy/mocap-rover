// A minimal HTTP/1.1 server for the node's settings page: one thread accepts,
// each request is answered on its own short-lived thread, Connection: close.
// Only what the page needs: method, path, query, body; no keep-alive, no TLS.
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>

namespace mocap {

struct HttpRequest {
    std::string method, path;
    std::map<std::string, std::string> query;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

class HttpServer {
public:
    using Handler = std::function<HttpResponse(const HttpRequest&)>;

    HttpServer(int port, Handler handler);
    ~HttpServer();

private:
    void accept_loop();
    void serve(int fd);

    int port_;
    Handler handler_;
    int listen_fd_ = -1;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace mocap
