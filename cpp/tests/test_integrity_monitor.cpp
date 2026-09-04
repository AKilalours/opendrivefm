// Parity + latency test for the C++ integrity monitor.
//
// Reads a binary dump written by scripts/dump_integrity_case.py from a real
// nuScenes keyframe: six camera calibrations, the occupancy grid at 0.5 m, and
// the reference maps the Python produced. Recomputes everything in C++ and
// reports the maximum absolute difference. Nothing here is a mock -- if the
// dump is missing the test says so and skips rather than passing vacuously.
#include "odfm/integrity_monitor.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace odfm;

template <typename T>
static bool rd(std::ifstream& f, T* p, size_t n) {
  f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n * sizeof(T)));
  return static_cast<bool>(f);
}

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "integrity_case.bin";
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::printf("SKIP: %s not found (run scripts/dump_integrity_case.py)\n", path.c_str());
    return 0;
  }
  int n = 0;
  double rng = 0, res = 0, full_px = 0, trust = 0;
  if (!rd(f, &n, 1) || !rd(f, &rng, 1) || !rd(f, &res, 1) || !rd(f, &full_px, 1) ||
      !rd(f, &trust, 1)) {
    std::printf("FAIL: short header\n");
    return 1;
  }
  GridSpec g{n, rng, res};
  const size_t M = static_cast<size_t>(n) * n;

  std::vector<CameraCalib> cams(6);
  for (auto& c : cams) {
    rd(f, c.quat.data(), 4);
    rd(f, c.trans.data(), 3);
    rd(f, c.K.data(), 9);
    rd(f, &c.width, 1);
    rd(f, &c.height, 1);
  }
  std::vector<unsigned char> occ(M);
  rd(f, occ.data(), M);
  std::vector<float> ref_integ(M);
  rd(f, ref_integ.data(), M);
  std::vector<std::vector<float>> ref_cov(6, std::vector<float>(M));
  for (auto& r : ref_cov) rd(f, r.data(), M);
  if (!f) {
    std::printf("FAIL: short dump\n");
    return 1;
  }

  std::vector<std::vector<float>> cov(6);
  std::vector<unsigned char> vis;
  std::vector<float> integ;
  std::vector<double> laps;
  const int ITERS = 25;
  for (int it = 0; it < ITERS + 2; ++it) {  // two warm-up passes
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 6; ++i) {
      ground_coverage(cams[static_cast<size_t>(i)], g, cov[static_cast<size_t>(i)], 0.0, full_px);
      visibility_from(occ, g, cams[static_cast<size_t>(i)].trans[0],
                      cams[static_cast<size_t>(i)].trans[1], vis);
      for (size_t k = 0; k < M; ++k) cov[static_cast<size_t>(i)][k] *= vis[k];
    }
    integrity_map(cov, std::vector<double>(6, trust), integ);
    if (it >= 2)
      laps.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
  }
  std::sort(laps.begin(), laps.end());
  const double ms = laps[laps.size() / 2];
  const double ms99 = laps[static_cast<size_t>(0.99 * (laps.size() - 1))];

  double worst_cov = 0;
  for (int i = 0; i < 6; ++i)
    for (size_t k = 0; k < M; ++k)
      worst_cov = std::max(worst_cov, static_cast<double>(std::fabs(
                                          cov[static_cast<size_t>(i)][k] - ref_cov[static_cast<size_t>(i)][k])));
  double worst_int = 0;
  for (size_t k = 0; k < M; ++k)
    worst_int = std::max(worst_int, static_cast<double>(std::fabs(integ[k] - ref_integ[k])));

  const auto st = integrity_stats(integ, {});
  std::printf("grid %dx%d @ %.2f m  |  6 cameras\n", n, n, res);
  std::printf("coverage x visibility  max abs diff  %.3e\n", worst_cov);
  std::printf("integrity map          max abs diff  %.3e\n", worst_int);
  std::printf("mean integrity  C++ %.6f\n", st.mean);
  std::printf("recompute latency  p50 %.2f ms  p99 %.2f ms  (%d iters, single thread)\n",
              ms, ms99, ITERS);

  // float32 storage of a value in [0,1] carries ~6e-8; 1e-5 leaves room for
  // the double-vs-float accumulation order differing between the two
  // implementations without letting a real algorithmic divergence through.
  const bool ok = worst_cov < 1e-5 && worst_int < 1e-5;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
