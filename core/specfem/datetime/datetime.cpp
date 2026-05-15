#include "specfem/datetime.hpp"

#include <charconv>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace specfem {
namespace datetime {

type make(int year, int month, int day, int hour, int minute, double second) {
  using namespace std::chrono;

  auto ymd = std::chrono::year{ year } /
             std::chrono::month{ static_cast<unsigned>(month) } /
             std::chrono::day{ static_cast<unsigned>(day) };

  if (!ymd.ok()) {
    throw std::runtime_error("Invalid date: " + std::to_string(year) + "-" +
                             std::to_string(month) + "-" + std::to_string(day));
  }

  auto dp = sys_days{ ymd };

  // Split second into integer and fractional parts
  int sec_int = static_cast<int>(second);
  int ms = static_cast<int>(std::round((second - sec_int) * 1000.0));

  auto tp = dp + hours{ hour } + minutes{ minute } + seconds{ sec_int } +
            milliseconds{ ms };

  return time_point_cast<milliseconds>(tp);
}

type parse_iso(const std::string &str) {
  // Accepts "YYYY-MM-DDTHH:MM:SS.ss" or "YYYY-MM-DD HH:MM:SS.ss"
  // Fractional seconds are optional.
  if (str.size() < 19) {
    throw std::runtime_error("Invalid datetime string (too short): \"" + str +
                             "\"");
  }

  auto parse_int = [&](int pos, int len) -> int {
    int val = 0;
    auto [ptr, ec] =
        std::from_chars(str.data() + pos, str.data() + pos + len, val);
    if (ec != std::errc{}) {
      throw std::runtime_error("Failed to parse integer at position " +
                               std::to_string(pos) + " in datetime string: \"" +
                               str + "\"");
    }
    return val;
  };

  int year = parse_int(0, 4);
  int month = parse_int(5, 2);
  int day = parse_int(8, 2);
  int hour = parse_int(11, 2);
  int minute = parse_int(14, 2);

  // Parse seconds (may include fractional part)
  double second = 0.0;
  {
    const char *sec_start = str.data() + 17;
    // Use strtod for fractional seconds
    char *end = nullptr;
    second = std::strtod(sec_start, &end);
    if (end == sec_start) {
      throw std::runtime_error(
          "Failed to parse seconds in datetime string: \"" + str + "\"");
    }
  }

  return make(year, month, day, hour, minute, second);
}

std::string to_string(const type &t) {
  using namespace std::chrono;

  auto dp = floor<days>(t);
  year_month_day ymd{ dp };
  auto time_of_day = t - dp;

  auto h = duration_cast<hours>(time_of_day);
  auto m = duration_cast<minutes>(time_of_day - h);
  auto s = duration_cast<seconds>(time_of_day - h - m);
  auto ms = duration_cast<milliseconds>(time_of_day - h - m - s);

  std::ostringstream os;
  os << std::setfill('0') << std::setw(4) << static_cast<int>(ymd.year()) << '-'
     << std::setw(2) << static_cast<unsigned>(ymd.month()) << '-'
     << std::setw(2) << static_cast<unsigned>(ymd.day()) << 'T' << std::setw(2)
     << h.count() << ':' << std::setw(2) << m.count() << ':' << std::setw(2)
     << s.count() << '.' << std::setw(3) << ms.count();

  return os.str();
}

} // namespace datetime
} // namespace specfem
