#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace kaptia {
struct DiagnosticResult {
    bool ok{false};
    unsigned long native_error{0};
    unsigned long http_status{0};
    std::wstring message;
    std::filesystem::path log_path;
};

std::filesystem::path log_path();
void log_line(std::wstring_view message);
void log_error(std::wstring_view operation, unsigned long code);
std::wstring win32_message(unsigned long code);
}
