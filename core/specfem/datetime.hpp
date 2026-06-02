#pragma once

#include <chrono>
#include <string>

namespace specfem {
namespace datetime {

/// UTC timestamp with millisecond precision
using type = std::chrono::sys_time<std::chrono::milliseconds>;

/// Build a datetime from calendar components (year, month, day, hour, minute,
/// second). The second parameter accepts fractional seconds (e.g. 52.40).
type make(int year, int month, int day, int hour, int minute, double second);

/// Parse an ISO 8601 datetime string "YYYY-MM-DDTHH:MM:SS.ss" into a datetime.
/// The 'T' separator may also be a space.
type parse_iso(const std::string &str);

/// Convert a datetime to a human-readable ISO 8601 string for logging.
std::string to_string(const type &t);

} // namespace datetime
} // namespace specfem
