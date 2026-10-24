#include "diagnostics.h"
#include "downloader.h"

#include <Windows.h>
#include <filesystem>
#include <iostream>
#include <string>

int wmain(int argc, wchar_t** argv) {
    using namespace kaptia;
    log_line(L"Kaptia native bootstrap diagnostic started.");

    const auto root = std::filesystem::path(_wgetenv(L"LOCALAPPDATA") ? _wgetenv(L"LOCALAPPDATA") : L".") / L"Kaptia";
    if (argc > 1 && std::wstring_view(argv[1]) == L"--test-download") {
        const auto target = root / L"downloads" / L"kaptia-source.zip";
        const auto result = download_https(
            L"https://github.com/argusocean-wq/kaptia/archive/refs/heads/main.zip", target);
        std::wcout << (result.ok ? L"SUCCESS: " : L"FAILED: ") << result.message << L"\n";
        std::wcout << L"HTTP status: " << result.http_status << L"\n";
        std::wcout << L"Log: " << result.log_path.wstring() << L"\n";
        return result.ok ? 0 : 1;
    }

    std::wcout << L"Kaptia C++ native bootstrap foundation\n"
               << L"This development build is not yet the complete Kaptia application.\n"
               << L"Run with --test-download to test the GitHub source download.\n"
               << L"Diagnostic log: " << log_path().wstring() << L"\n";
    log_line(L"No operation requested. Use --test-download for a network diagnostic.");
    return 0;
}
