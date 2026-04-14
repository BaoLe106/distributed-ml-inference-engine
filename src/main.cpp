#include <iostream>
#include <memory>
#include <map>
#include <optional>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "gateway.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <worker1:port> [worker2:port] ..." << std::endl;
        std::cerr << "Example: " << argv[0] << " localhost:8001 localhost:8002 localhost:8003" << std::endl;
        return 1;
    }
    
    std::vector<std::string> workers;
    for (int i = 1; i < argc; ++i) {
        workers.push_back(argv[i]);
    }
    
    Gateway gateway(workers);
    httplib::Server server;
    // inference endpoint
    server.Post("/infer", [&gateway](const httplib::Request& req, httplib::Response& res) {
        try {
            auto request = json::parse(req.body);
            auto response = gateway.routeRequest(request);
            
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error;
            error["error"] = e.what();
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    });
    // stats endpoint
    server.Get("/stats", [&gateway](const httplib::Request&, httplib::Response& res) {
        auto stats = gateway.getStats();
        res.set_content(stats.dump(), "application/json");
    });
    std::cout << "Gateway listening on port 8000" << std::endl;
    std::cout << "Workers: " << workers.size() << std::endl;
    std::cout << "Circuit breakers enabled" << std::endl;
    std::cout << "Ready!" << std::endl;
    server.listen("0.0.0.0", 8000);
    return 0;
}