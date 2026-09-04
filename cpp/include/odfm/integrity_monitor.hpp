// Perception integrity (camera observability) monitor -- C++ port.
//
// The Python implementation lives in scripts/odfm_geom.py and
// scripts/odfm_ground.py. This is the deployment-side copy: header-only, no
// LibTorch, no allocation inside the hot loop after the grids are sized. It is
// the piece the runner needs that the model does not provide -- a monitor that
// answers "how much of the drivable ground is covered by cameras I trust,
// right now", from calibration and an occupancy grid alone.
//
// Three stages, matching the Python exactly:
//
//   1. ground_coverage(cam)  per-cell projected AREA of the cell quad in the
//      camera image, normalised by full_px and clipped to [0,1]. Area, not a
//      binary frustum test: a cell at 54 m is viewed at grazing incidence and
//      images to a sliver, and must not count as evidence equal to a cell at
//      5 m.
//   2. visibility_from(occ, origin)  a bearing-bucket ray cast from the
//      CAMERA position (not the LiDAR), so occlusion shadows land where the
//      camera actually is. A running minimum over 3 neighbouring bearings
//      closes the sliver gaps a one-cell-wide obstacle would otherwise leave.
//   3. integrity = 1 - prod_i (1 - trust_i * coverage_i * visible_i)
//      noisy-OR of independent evidence. The independence is an upper bound,
//      not an unbiased estimate: cameras on one vehicle share weather and sun.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace odfm {

struct CameraCalib {
  std::array<double, 4> quat{{1, 0, 0, 0}};  // (w, x, y, z), sensor->ego
  std::array<double, 3> trans{{0, 0, 0}};    // sensor origin in ego frame
  std::array<double, 9> K{};                 // row-major 3x3 intrinsic
  double width = 1600.0, height = 900.0;
};

struct GridSpec {
  int n = 216;
  double rng_m = 54.0;
  double res = 0.5;
};

// (w,x,y,z) quaternion -> row-major 3x3 rotation, identical to
// odfm_tables.quat_to_rot.
inline std::array<double, 9> quat_to_rot(const std::array<double, 4>& q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
           2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
           2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)}};
}

// Ego-frame point -> pixel. Mirrors odfm_geom.project_to_image, including its
// guard: a pinhole model produces finite pixel coordinates for points BEHIND
// the camera, which is the classic way an overlay mirrors objects into the sky.
struct Projected {
  double u = 0, v = 0, depth = 0;
  bool valid = false;
};

inline Projected project(const CameraCalib& c, const std::array<double, 9>& R,
                         double x, double y, double z, double min_depth = 0.5) {
  const double dx = x - c.trans[0], dy = y - c.trans[1], dz = z - c.trans[2];
  // (p - t) @ R  -- R applied on the right, as in the Python
  const double cx = dx * R[0] + dy * R[3] + dz * R[6];
  const double cy = dx * R[1] + dy * R[4] + dz * R[7];
  const double cz = dx * R[2] + dy * R[5] + dz * R[8];
  const double safe = (std::abs(cz) < 1e-9) ? 1e-9 : cz;
  const double uw = cx * c.K[0] + cy * c.K[1] + cz * c.K[2];
  const double vw = cx * c.K[3] + cy * c.K[4] + cz * c.K[5];
  Projected p;
  p.u = uw / safe;
  p.v = vw / safe;
  p.depth = cz;
  p.valid = cz > min_depth && p.u >= 0 && p.u < c.width && p.v >= 0 && p.v < c.height;
  return p;
}

// Stage 1. out is resized to n*n, row-major with index [ix * n + iy], matching
// numpy's meshgrid(indexing="ij") layout.
inline void ground_coverage(const CameraCalib& c, const GridSpec& g,
                            std::vector<float>& out, double z_ground = 0.0,
                            double full_px = 140.0) {
  const auto R = quat_to_rot(c.quat);
  const int n = g.n;
  out.assign(static_cast<size_t>(n) * n, 0.f);
  const double h = g.res / 2.0;
  static const double dxs[4] = {-1, 1, 1, -1};
  static const double dys[4] = {-1, -1, 1, 1};
  for (int ix = 0; ix < n; ++ix) {
    const double X = (ix + 0.5) * g.res - g.rng_m;
    for (int iy = 0; iy < n; ++iy) {
      const double Y = (iy + 0.5) * g.res - g.rng_m;
      if (!project(c, R, X, Y, z_ground).valid) continue;
      double u[4], v[4];
      for (int k = 0; k < 4; ++k) {
        const Projected p = project(c, R, X + dxs[k] * h, Y + dys[k] * h, z_ground);
        u[k] = p.u;
        v[k] = p.v;
      }
      double s = 0;  // shoelace
      for (int k = 0; k < 4; ++k) {
        const int m = (k + 1) & 3;
        s += u[k] * v[m] - u[m] * v[k];
      }
      const double area = 0.5 * std::abs(s);
      out[static_cast<size_t>(ix) * n + iy] =
          static_cast<float>(std::min(1.0, std::max(0.0, area / full_px)));
    }
  }
}

// Stage 2. occ is the same row-major n*n grid, non-zero where occupied.
inline void visibility_from(const std::vector<unsigned char>& occ, const GridSpec& g,
                            double ox, double oy, std::vector<unsigned char>& out,
                            int n_azimuth = 2048, double slack = 1.0) {
  const int n = g.n;
  out.assign(static_cast<size_t>(n) * n, 1);
  std::vector<double> horizon(static_cast<size_t>(n_azimuth),
                              std::numeric_limits<double>::infinity());
  const double TWO_PI = 6.283185307179586;
  auto bearing = [&](double dx, double dy) {
    int b = static_cast<int>((std::atan2(dy, dx) + M_PI) / TWO_PI * n_azimuth);
    return std::min(std::max(b, 0), n_azimuth - 1);
  };
  bool any = false;
  for (int ix = 0; ix < n; ++ix) {
    const double dx = (ix + 0.5) * g.res - g.rng_m - ox;
    for (int iy = 0; iy < n; ++iy) {
      if (!occ[static_cast<size_t>(ix) * n + iy]) continue;
      any = true;
      const double dy = (iy + 0.5) * g.res - g.rng_m - oy;
      const double r = std::hypot(dx, dy);
      double& hh = horizon[static_cast<size_t>(bearing(dx, dy))];
      if (r < hh) hh = r;
    }
  }
  if (!any) return;
  // Running minimum over 3 neighbouring bearings, wrapping -- what a solid
  // object actually does to the light.
  std::vector<double> sm(horizon.size());
  for (int b = 0; b < n_azimuth; ++b) {
    const int lo = (b - 1 + n_azimuth) % n_azimuth, hi = (b + 1) % n_azimuth;
    sm[static_cast<size_t>(b)] =
        std::min(horizon[static_cast<size_t>(lo)],
                 std::min(horizon[static_cast<size_t>(b)], horizon[static_cast<size_t>(hi)]));
  }
  for (int ix = 0; ix < n; ++ix) {
    const double dx = (ix + 0.5) * g.res - g.rng_m - ox;
    for (int iy = 0; iy < n; ++iy) {
      const double dy = (iy + 0.5) * g.res - g.rng_m - oy;
      const double r = std::hypot(dx, dy);
      out[static_cast<size_t>(ix) * n + iy] =
          r <= sm[static_cast<size_t>(bearing(dx, dy))] + slack * g.res ? 1 : 0;
    }
  }
}

// Stage 3.
inline void integrity_map(const std::vector<std::vector<float>>& coverage,
                          const std::vector<double>& trust, std::vector<float>& out) {
  const size_t m = coverage.empty() ? 0 : coverage[0].size();
  out.assign(m, 0.f);
  for (size_t i = 0; i < m; ++i) {
    double acc = 1.0;
    for (size_t c = 0; c < coverage.size(); ++c)
      acc *= 1.0 - trust[c] * static_cast<double>(coverage[c][i]);
    out[i] = static_cast<float>(1.0 - acc);
  }
}

struct IntegrityStats {
  double mean = 0, frac_below_0_3 = 0, frac_above_0_7 = 0, mean_over_free = 0;
  size_t free_cells = 0;
};

// `free_mask` is optional (empty = whole grid). Mean over the whole square is
// dominated by cells behind the vehicle that no planner will route through;
// mean over free space is the number that answers the operational question.
inline IntegrityStats integrity_stats(const std::vector<float>& integ,
                                      const std::vector<unsigned char>& free_mask) {
  IntegrityStats s;
  double sum = 0, fsum = 0;
  size_t lo = 0, hi = 0;
  for (size_t i = 0; i < integ.size(); ++i) {
    sum += integ[i];
    if (integ[i] < 0.3) ++lo;
    if (integ[i] > 0.7) ++hi;
    if (!free_mask.empty() && free_mask[i]) {
      fsum += integ[i];
      ++s.free_cells;
    }
  }
  const double n = static_cast<double>(integ.size());
  s.mean = n ? sum / n : 0.0;
  s.frac_below_0_3 = n ? lo / n : 0.0;
  s.frac_above_0_7 = n ? hi / n : 0.0;
  s.mean_over_free = s.free_cells ? fsum / static_cast<double>(s.free_cells) : 0.0;
  return s;
}

}  // namespace odfm
