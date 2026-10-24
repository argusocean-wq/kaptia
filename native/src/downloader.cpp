#include "downloader.h"

#include <Windows.h>
#include <winhttp.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#pragma comment(lib, "winhttp.lib")

namespace {
struct InternetHandle {
    HINTERNET value{};
    explicit InternetHandle(HINTERNET handle = nullptr) : value(handle) {}
    ~InternetHandle() { if (value) WinHttpCloseHandle(value); }
    InternetHandle(const InternetHandle&) = delete;
    InternetHandle& operator=(const InternetHandle&) = delete;
};

std::wstring system_error_text(DWORD code) {
    return L"WinHTTP/Win32 error " + std::to_wstring(code) + L": " + kaptia::win32_message(code);
}
}

namespace kaptia {
DiagnosticResult download_https(std::wstring_view url, const std::filesystem::path& destination) {
    DiagnosticResult result{};
    result.log_path = log_path();
    const std::wstring url_copy(url);
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwSchemeLength = static_cast<DWORD>(-1);
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    parts.dwExtraInfoLength = static_cast<DWORD>(-1);

    if (!WinHttpCrackUrl(url_copy.c_str(), static_cast<DWORD>(url_copy.size()), 0, &parts)) {
        result.native_error = GetLastError();
        result.message = L"URL could not be parsed. Check the URL format.";
        log_error(L"WinHttpCrackUrl", result.native_error);
        return result;
    }
    if (parts.nScheme != INTERNET_SCHEME_HTTPS || parts.dwHostNameLength == 0) {
        result.message = L"Download blocked: only valid HTTPS URLs are accepted.";
        log_line(result.message);
        return result;
    }

    const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    if (parts.dwExtraInfoLength) path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);
    if (path.empty()) path = L"/";

    InternetHandle session(WinHttpOpen(L"Kaptia-Native/0.1",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session.value) {
        result.native_error = GetLastError();
        result.message = system_error_text(result.native_error);
        log_error(L"WinHttpOpen", result.native_error);
        return result;
    }
    WinHttpSetTimeouts(session.value, 15000, 15000, 30000, 30000);

    InternetHandle connection(WinHttpConnect(session.value, host.c_str(), parts.nPort, 0));
    if (!connection.value) {
        result.native_error = GetLastError();
        result.message = system_error_text(result.native_error);
        log_error(L"WinHttpConnect", result.native_error);
        return result;
    }

    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(session.value, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy, sizeof(redirect_policy));
    InternetHandle request(WinHttpOpenRequest(connection.value, L"GET", path.c_str(),
        nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        WINHTTP_FLAG_SECURE));
    if (!request.value) {
        result.native_error = GetLastError();
        result.message = system_error_text(result.native_error);
        log_error(L"WinHttpOpenRequest", result.native_error);
        return result;
    }

    if (!WinHttpSendRequest(request.value, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        result.native_error = GetLastError();
        result.message = system_error_text(result.native_error);
        log_error(L"WinHttpSendRequest", result.native_error);
        return result;
    }
    if (!WinHttpReceiveResponse(request.value, nullptr)) {
        result.native_error = GetLastError();
        result.message = system_error_text(result.native_error);
        log_error(L"WinHttpReceiveResponse", result.native_error);
        return result;
    }

    DWORD status = 0;
    DWORD status_size = sizeof(status);
    if (!WinHttpQueryHeaders(request.value, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size, WINHTTP_NO_HEADER_INDEX)) {
        result.native_error = GetLastError();
        result.message = system_error_text(result.native_error);
        log_error(L"WinHttpQueryHeaders(status)", result.native_error);
        return result;
    }
    result.http_status = status;
    log_line(L"HTTP response status: " + std::to_wstring(status) + L" for " + host + path);
    if (status < 200 || status >= 300) {
        result.message = L"Download server returned HTTP " + std::to_wstring(status) +
            L". Check the release URL, network filter, or GitHub availability.";
        log_line(result.message);
        return result;
    }

    std::error_code ec;
    std::filesystem::create_directories(destination.parent_path(), ec);
    if (ec) {
        result.native_error = ec.value();
        result.message = L"Cannot create download folder: " + destination.parent_path().wstring();
        log_line(result.message + L"; error=" + std::to_wstring(result.native_error));
        return result;
    }

    auto partial = destination;
    partial += L".part";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output) {
        result.message = L"Cannot create temporary download file. Check folder permissions and free disk space.";
        log_line(result.message + L" Path: " + partial.wstring());
        return result;
    }

    unsigned long long total = 0;
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.value, &available)) {
            result.native_error = GetLastError();
            result.message = system_error_text(result.native_error);
            log_error(L"WinHttpQueryDataAvailable", result.native_error);
            output.close();
            std::filesystem::remove(partial, ec);
            return result;
        }
        if (available == 0) break;
        std::vector<char> buffer(available);
        DWORD received = 0;
        if (!WinHttpReadData(request.value, buffer.data(), available, &received)) {
            result.native_error = GetLastError();
            result.message = system_error_text(result.native_error);
            log_error(L"WinHttpReadData", result.native_error);
            output.close();
            std::filesystem::remove(partial, ec);
            return result;
        }
        if (received == 0) break;
        output.write(buffer.data(), static_cast<std::streamsize>(received));
        if (!output) {
            result.message = L"Disk write failed while downloading. Check free space and permissions.";
            log_line(result.message);
            output.close();
            std::filesystem::remove(partial, ec);
            return result;
        }
        total += received;
    }
    output.flush();
    if (!output) {
        result.message = L"Download could not be flushed to disk.";
        log_line(result.message);
        output.close();
        std::filesystem::remove(partial, ec);
        return result;
    }
    output.close();

    if (total == 0) {
        result.message = L"Server returned an empty download.";
        log_line(result.message);
        std::filesystem::remove(partial, ec);
        return result;
    }
    std::filesystem::rename(partial, destination, ec);
    if (ec) {
        result.native_error = ec.value();
        result.message = L"Downloaded data is saved as a partial file but could not be finalized: " + ec.message().c_str();
        log_line(result.message);
        return result;
    }
    result.ok = true;
    result.message = L"Download completed: " + std::to_wstring(total) + L" bytes.";
    log_line(result.message + L" Destination: " + destination.wstring());
    return result;
}
}
