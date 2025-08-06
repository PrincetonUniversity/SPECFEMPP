#pragma once

namespace specfem::connections {

enum class type : int { strongly_conforming = 1 };

enum class orientation : int {
  top,
  right,
  bottom,
  left,
  top_left,
  top_right,
  bottom_right,
  bottom_left
};

const std::list<orientation> edges = { orientation::top, orientation::right,
                                       orientation::bottom, orientation::left };

const std::list<orientation> corners = { orientation::top_left,
                                         orientation::top_right,
                                         orientation::bottom_right,
                                         orientation::bottom_left };

template <typename T> bool contains(const T &list, const orientation &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

std::list<orientation> edges_of_corner(const orientation &corner);

class connection_mapping {
public:
  connection_mapping(const int ngllx, const int ngllz);

  std::tuple<std::tuple<int, int>, std::tuple<int, int> >
  map_coordinates(const orientation &from, const orientation &to,
                  const int point) const;

  int find_corner_on_edge(const orientation &corner,
                          const orientation &edge) const;

private:
  int ngllx, ngllz;

  // corner coordinates
  std::unordered_map<orientation, std::tuple<int, int> > coordinates_at_corner;
  std::unordered_map<orientation, std::function<std::tuple<int, int>(int)> >
      coordinates_along_edge;
};

} // namespace specfem::connections
