#!/bin/sh
# Every check that needs neither a GPU nor the nuScenes export.
# Exits non-zero the moment one fails; there is no summary-only mode.
#
# ODFM_HOME lets this run against a checkout outside a container, which is how
# it was validated -- see A39. Inside the image it is /odfm.
set -eu
cd "${ODFM_HOME:-/odfm}"
what="${1:-all}"

run_cpp() {
  echo "== C++ runtime primitives, release build"
  ctest --test-dir cpp/build --output-on-failure
  echo "== C++ runtime primitives, ThreadSanitizer"
  ctest --test-dir cpp/build-tsan --output-on-failure
}

run_tests() {
  echo "== Test suite collects (a suite that cannot collect is a gate that cannot fail)"
  python -m pytest tests/ --collect-only -q
  echo "== Test suite"
  python -m pytest tests/ -q
}

run_gates() {
  echo "== Release gates on the committed detection artifact"
  python scripts/ci/check_gates.py \
      --detection outputs/artifacts/ood_detection_report_v11_trustfix2.json

  echo "== Gates must REJECT the known-inverted v11 baseline"
  if python scripts/ci/check_gates.py \
       --detection outputs/artifacts/ood_detection_report_v11.json >/dev/null 2>&1; then
    echo "FAIL: gates passed on the known-bad baseline; they have been weakened." >&2
    exit 1
  fi
  echo "   gates correctly reject it"

  echo "== Observability gates (A12-A38)"
  python scripts/ci/check_observability_gates.py
  echo "== Observability gates must reject deliberately broken artifacts"
  python scripts/ci/check_observability_gates.py --self-test
}

case "$what" in
  cpp)   run_cpp ;;
  tests) run_tests ;;
  gates) run_gates ;;
  all)   run_cpp; run_tests; run_gates
         echo; echo "All checks passed." ;;
  *)     exec "$@" ;;
esac
