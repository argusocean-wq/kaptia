#pragma once

#include "diagnostics.h"
#include <filesystem>
#include <string_view>

namespace kaptia {
DiagnosticResult download_https(std::wstring_view url, const std::filesystem::path& destination);
}
