#include "http_server.hpp"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

namespace mocap {

namespace {

std::string url_decode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            out += char(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += s[i] == '+' ? ' ' : s[i];
        }
    }
    return out;
}

const char* reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        default: return "Error";
    }
}

}  // namespace

HttpServer::HttpServer(int port, Handler handler) : port_(port), handler_(std::move(handler)) {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(uint16_t(port));
    a.sin_addr.s_addr = INADDR_ANY;
    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) || listen(listen_fd_, 8)) {
        std::cerr << "http: cannot listen on " << port << ": " << std::strerror(errno) << "\n";
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }
    thread_ = std::thread(&HttpServer::accept_loop, this);
}

HttpServer::~HttpServer() {
    stop_ = true;
    if (listen_fd_ >= 0) shutdown(listen_fd_, SHUT_RDWR);
    if (thread_.joinable()) thread_.join();
    if (listen_fd_ >= 0) close(listen_fd_);
}

void HttpServer::accept_loop() {
    while (!stop_) {
        int fd = accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        timeval tv{5, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        std::thread(&HttpServer::serve, this, fd).detach();
    }
}

void HttpServer::serve(int fd) {
    std::string data;
    char buf[4096];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos && data.size() < 65536) {
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        data.append(buf, size_t(n));
        header_end = data.find("\r\n\r\n");
    }
    if (header_end == std::string::npos) {
        close(fd);
        return;
    }
    HttpRequest req;
    std::istringstream head(data.substr(0, header_end));
    std::string target, version, line;
    head >> req.method >> target >> version;
    size_t content_length = 0;
    std::getline(head, line);
    while (std::getline(head, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        for (auto& c : key) c = char(std::tolower(c));
        if (key == "content-length") content_length = std::stoul(line.substr(colon + 1));
    }
    req.body = data.substr(header_end + 4);
    while (req.body.size() < content_length) {
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        req.body.append(buf, size_t(n));
    }
    auto q = target.find('?');
    req.path = target.substr(0, q);
    if (q != std::string::npos) {
        std::istringstream qs(target.substr(q + 1));
        std::string kv;
        while (std::getline(qs, kv, '&')) {
            auto eq = kv.find('=');
            req.query[url_decode(kv.substr(0, eq))] = eq == std::string::npos ? "" : url_decode(kv.substr(eq + 1));
        }
    }
    HttpResponse res;
    try {
        res = handler_(req);
    } catch (const std::exception& e) {
        res.status = 400;
        res.body = std::string("{\"error\":\"") + e.what() + "\"}";
    }
    std::ostringstream out;
    out << "HTTP/1.1 " << res.status << " " << reason(res.status) << "\r\n"
        << "Content-Type: " << res.content_type << "\r\n"
        << "Content-Length: " << res.body.size() << "\r\n"
        << "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    std::string head_out = out.str();
    send(fd, head_out.data(), head_out.size(), MSG_NOSIGNAL);
    size_t off = 0;
    while (off < res.body.size()) {
        ssize_t w = send(fd, res.body.data() + off, res.body.size() - off, MSG_NOSIGNAL);
        if (w <= 0) break;
        off += size_t(w);
    }
    close(fd);
}

}  // namespace mocap
