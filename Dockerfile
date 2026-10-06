# OpenDriveFM -- the verification image.
#
# WHAT THIS IS
#   One command that reproduces every check this repository can run without a
#   GPU and without the 26 GB of nuScenes data: the C++ runtime primitives
#   under ThreadSanitizer, the Python test suite, and the release gates on the
#   committed artifacts, including the self-test that requires each gate to
#   FAIL on a deliberately broken input.
#
#       git archive HEAD | docker build -t odfm -
#       docker run --rm odfm            # everything, ~2 min
#       docker run --rm odfm gates      # just the gates
#       docker run --rm odfm tests      # just pytest
#
# WHAT THIS IS NOT
#   It does not reproduce the measurements. Rebuilding the 6,019 observability
#   maps needs the packed nuScenes export (26 GB, not redistributable) and
#   re-running FB-OCC needs a GPU. Mount the data and it will run:
#
#       docker run --rm -v /path/to/data:/odfm/data odfm \
#           python scripts/eval/safety_envelope.py --report
#
#   Stating that boundary is the point. An image that claimed to reproduce
#   numbers it cannot reach would be worse than no image.
#
# WHY `git archive` AND NOT `docker build .`
#   The working tree carries 26 GB of packed nuScenes and 4 GB of checkpoints.
#   Building from the committed tree makes the image a function of a commit
#   hash and nothing else, which is the property that makes it worth shipping.
#   .dockerignore reproduces the same result from a dirty tree if you prefer.
#
# WHY THE TORCH SPLIT
#   `pip install torch` pulls the CUDA build: ~2.5 GB of wheels for a runner
#   with no GPU. The CPU index is a separate `--index-url` command because
#   --index-url REPLACES PyPI rather than adding to it, so pytest cannot be
#   installed in the same breath (it does not exist on download.pytorch.org).
#   This is the same trap documented in .github/workflows/validation.yml.

# ---------------------------------------------------------------- cpp stage
FROM debian:bookworm-slim AS cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# A51: build at the SAME path the runtime image uses. CMake bakes absolute
# paths into CTestTestfile.cmake, so binaries built at /build/out and copied
# to /odfm/cpp/build are still looked for at /build/out, and every ctest
# reports "Unable to find executable". Caught by the CI job added in A39,
# which is the first time that image was ever actually built and run.
WORKDIR /odfm
COPY cpp/ cpp/

# Release build plus the TSan build. A lock-free SPSC queue and a seqlock can
# pass a functional test and still be racy; TSan is what checks the orderings.
RUN cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build cpp/build -j"$(nproc)" \
 && cmake -S cpp -B cpp/build-tsan -DCMAKE_BUILD_TYPE=Debug -DODFM_TSAN=ON \
 && cmake --build cpp/build-tsan -j"$(nproc)"

# ---------------------------------------------------------------- runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /odfm

# torch first, from the CPU index, on its own layer so the rest can change
# without re-downloading 200 MB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

# Only what the checks read. Not data/, not outputs/artifacts/*.npy, not the
# 26 GB export -- see the note at the top.
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY outputs/artifacts/ outputs/artifacts/
COPY --from=cpp /odfm/cpp/build      cpp/build/
COPY --from=cpp /odfm/cpp/build-tsan cpp/build-tsan/

COPY docker/verify.sh /usr/local/bin/verify
RUN chmod +x /usr/local/bin/verify

ENV PYTHONPATH=/odfm/src:/odfm
ENTRYPOINT ["verify"]
CMD ["all"]
