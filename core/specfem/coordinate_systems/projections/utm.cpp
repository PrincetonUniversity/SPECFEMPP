/// @brief UTM forward/inverse projection via the PROJ library.

#include "specfem/coordinate_systems/projections/utm.hpp"
#include <proj.h>
#include <string>

namespace specfem {
namespace coordinate_systems {

namespace {

// Build a normalized CRS-to-CRS transform for the given UTM zone.
// "Normalized" means geographic CRS has (longitude, latitude) axis order,
// matching our geographic_coordinates struct.
PJ *make_utm_transform(PJ_CONTEXT *ctx, int zone, bool south) {
  const int epsg = south ? (32700 + zone) : (32600 + zone);
  const std::string to = "EPSG:" + std::to_string(epsg);
  PJ *P = proj_create_crs_to_crs(ctx, "EPSG:4326", to.c_str(), nullptr);
  PJ *P_norm = proj_normalize_for_visualization(ctx, P);
  proj_destroy(P);
  return P_norm;
}

} // anonymous namespace

cartesian_coordinates to_cartesian(const geographic_coordinates &geo,
                                   const utm_projection_config &config) {
  if (config.suppress) {
    return { geo.longitude, geo.latitude, geo.depth };
  }

  PJ_CONTEXT *ctx = proj_context_create();
  PJ *P = make_utm_transform(ctx, std::abs(config.zone), config.zone < 0);

  PJ_COORD input = proj_coord(geo.longitude, geo.latitude, 0.0, 0.0);
  PJ_COORD output = proj_trans(P, PJ_FWD, input);

  proj_destroy(P);
  proj_context_destroy(ctx);

  return { output.xy.x, output.xy.y, geo.depth };
}

geographic_coordinates to_geographic(const cartesian_coordinates &cart,
                                     const utm_projection_config &config) {
  if (config.suppress) {
    return { cart.x, cart.y, cart.z };
  }

  PJ_CONTEXT *ctx = proj_context_create();
  PJ *P = make_utm_transform(ctx, std::abs(config.zone), config.zone < 0);

  PJ_COORD input = proj_coord(cart.x, cart.y, 0.0, 0.0);
  PJ_COORD output = proj_trans(P, PJ_INV, input);

  proj_destroy(P);
  proj_context_destroy(ctx);

  return { output.xy.x, output.xy.y, cart.z };
}

} // namespace coordinate_systems
} // namespace specfem
