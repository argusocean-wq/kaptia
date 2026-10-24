#include "diagnostics.h"

#include <Windows.h>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <system_error>

namespace {
std::mutex log_mutex;

std::filesystem::path app_data_root() {
    wchar_t* value = nullptr;
    size_t size = 0;
    if (_wdupenv_s(&value, &size, L"LOCALAPPDATA") == 0 && value != nullptr) {
        std::filesystem::path result(value);
        std::free(value);
        return result / L"Kaptia";
    }
    return std::filesystem::temp_directory_path() / L"Kaptia";
}
}

namespace kaptia {
std::filesystem::path log_path() {
    return app_data_root() / L"logs" / L"native-bootstrap.log";
}

void log_line(std::wstring_view message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    try {
        const auto path = log_path();
        std::filesystem::create_directories(path.parent_path());
        std::wofstream stream(path, std::ios::app);
        if (stream) {
            SYSTEMTIME now{};
            GetLocalTime(&now);
            stream << L'[' << now.wYear << L'-' << now.wMonth << L'-' << now.wDay
                   << L' ' << now.wHour << L':' << now.wMinute << L':' << now.wSecond
                   << L"] " << message << L'\n';
        }
    } catch (...) {
        // Logging must never hide the original failure.
    }
}

std::wstring win32_message(unsigned long code) {
    wchar_t* buffer = nullptr;
    const DWORD size = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, static_cast<DWORD>(code), 0,
        reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
    std::wstring result = size && buffer ? std::wstring(buffer, size) : L"No system message available";
    if (buffer) LocalFree(buffer);
    while (!result.empty() && (result.back() == L'\r' || result.back() == L'\n' || result.back() == L' ')) {
        result.pop_back();
    }
    return result;
}

void log_error(std::wstring_view operation, unsigned long code) {
    log_line(std::wstring(operation) + L" failed; Win32 error=" + std::to_wstring(code) +
             L"; " + win32_message(code));
}
}
