## Diagnosis

The immediate failure is **not yet an NPU, model, or BF16 problem**. ONNX Runtime cannot load the Vitis AI EP’s required shared object:

```text
/home/stack/repos/onnxruntime-build/Release/libonnxruntime_vitisai_ep.so
```

Your scripts have three likely root causes:

1. **`libonnxruntime_vitisai_ep.so` genuinely does not exist in the ORT directory used at runtime.**
2. **`run.sh` incorrectly treats two different libraries as interchangeable:**
   - `libonnxruntime_vitisai_ep.so`
   - `libonnxruntime_providers_vitisai.so`
3. **You are mixing components from different installations/releases:**
   - custom ONNX Runtime library from `/home/stack/repos/onnxruntime-build/Release`
   - headers from a source tree and `/usr/include`
   - Fedora-packaged ONNX Runtime development files
   - potentially a separate Ryzen AI/Vitis AI runtime

Those components must come from **one compatible AMD Ryzen AI release**. AMD also requires the Vitis AI EP, configuration file, drivers, and caches to stay version-compatible. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/1.3/app_development.html?utm_source=openai))

There is an additional platform concern: AMD’s current documented Linux configuration is **Ubuntu 24.04 LTS, kernel 6.10 or newer**, with STX/KRK support. The Ryzen AI 9 HX 370 is an STX device, but Fedora 44 is not AMD’s documented supported distribution. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/linux.html?utm_source=openai))

---

# 1. Remove the invalid symlink behavior immediately

This section of `run.sh` is unsafe:

```bash
if [ -z "$VITIS_EP" ]; then
    VITIS_EP="$(find "$AMD_ORT_LIB" -maxdepth 1 \
        -name 'libonnxruntime_providers_vitisai.so*' \
        -type f -print -quit)"
fi
```

It then does:

```bash
ln -sf "$(basename "$VITIS_EP")" \
    "${AMD_ORT_LIB}/libonnxruntime_vitisai_ep.so"
```

That can create:

```text
libonnxruntime_vitisai_ep.so -> libonnxruntime_providers_vitisai.so
```

These are **not alternate names for the same library**. Do not make that symlink.

Delete any invalid link:

```bash
rm -f /home/stack/repos/onnxruntime-build/Release/libonnxruntime_vitisai_ep.so
```

Then replace the discovery logic with:

```bash
VITIS_EP="$(find "$AMD_ORT_LIB" -maxdepth 1 \
    \( -type f -o -type l \) \
    -name 'libonnxruntime_vitisai_ep.so*' \
    -print -quit)"

if [ -z "$VITIS_EP" ]; then
    echo "[FATAL] The required libonnxruntime_vitisai_ep.so is absent from:"
    echo "  $AMD_ORT_LIB"
    echo
    echo "libonnxruntime_providers_vitisai.so is not a substitute."
    exit 1
fi
```

Only create an unversioned symlink when the target is another version of the **same library**:

```bash
if [ ! -e "$AMD_ORT_LIB/libonnxruntime_vitisai_ep.so" ]; then
    case "$(basename "$VITIS_EP")" in
        libonnxruntime_vitisai_ep.so.*)
            ln -s "$(basename "$VITIS_EP")" \
                "$AMD_ORT_LIB/libonnxruntime_vitisai_ep.so"
            ;;
        *)
            echo "[FATAL] Unexpected Vitis AI EP filename: $VITIS_EP"
            exit 1
            ;;
    esac
fi
```

---

# 2. Determine exactly which ONNX Runtime your executable loads

Run:

```bash
readelf -d ./puck_eye_npu_server | grep -E 'RPATH|RUNPATH|NEEDED'
ldd ./puck_eye_npu_server | grep -E 'onnxruntime|vitis|ryzen|xrt'
```

Given your build command, the expected result is:

```text
libonnxruntime.so => /home/stack/repos/onnxruntime-build/Release/libonnxruntime.so
```

Also check the ELF interpreter and architecture:

```bash
file ./puck_eye_npu_server
file /home/stack/repos/onnxruntime-build/Release/libonnxruntime.so
```

Then inspect the exact missing path:

```bash
EP=/home/stack/repos/onnxruntime-build/Release/libonnxruntime_vitisai_ep.so

ls -la "$EP"
stat "$EP"
readlink -e "$EP"
namei -l "$EP"
```

Interpretation:

- `ls: No such file` → the EP was never built or installed there.
- `readlink -e` prints nothing → broken symlink or inaccessible target.
- `namei` shows a missing directory component → path or mount problem.
- File exists but loading still fails → continue to dependency and ABI checks.

Use the dynamic loader directly:

```bash
LD_DEBUG=libs,files \
./puck_eye_npu_server puck-eye-seg-s_bf16_ctx.onnx 18888 \
2>&1 | tee loader.log
```

Then:

```bash
grep -E 'onnxruntime|vitisai|ryzenai|not found|error' loader.log
```

This is more authoritative than printing environment variables.

---

# 3. Check all required shared-library dependencies

If the exact EP exists:

```bash
EP=/home/stack/repos/onnxruntime-build/Release/libonnxruntime_vitisai_ep.so

file "$EP"
readelf -d "$EP" | grep -E 'NEEDED|RPATH|RUNPATH'
ldd "$EP"
```

Fail the test if anything is unresolved:

```bash
if ldd "$EP" | grep -q 'not found'; then
    echo "Unresolved dependencies:"
    ldd "$EP" | grep 'not found'
    exit 1
fi
```

Check all related libraries, not just that one:

```bash
find /home/stack/repos/onnxruntime-build/Release \
    -maxdepth 1 -type f -o -type l |
    grep -E 'onnxruntime|vitis|ryzen|xrt' |
    sort
```

Depending on the Ryzen AI/ORT generation, companion libraries may include provider-shared, Vitis AI provider, Ryzen AI provider, compiler/runtime, XRT, and dispatch libraries. Do not combine similarly named libraries from different releases.

A useful recursive audit:

```bash
for lib in /home/stack/repos/onnxruntime-build/Release/*.so*; do
    if file "$lib" | grep -q ELF; then
        missing="$(ldd "$lib" 2>/dev/null | grep 'not found' || true)"
        if [ -n "$missing" ]; then
            echo
            echo "### $lib"
            echo "$missing"
        fi
    fi
done
```

---

# 4. Stop mixing the Fedora ONNX Runtime with AMD ONNX Runtime

Your setup currently has multiple competing sources:

```text
/usr/include/onnxruntime
/home/stack/repos/onnxruntime-src/include/...
/home/stack/repos/onnxruntime-build/Release/libonnxruntime.so
Fedora onnxruntime-devel
Ryzen AI installation
```

That is fragile even if it compiles. A stock distribution ONNX Runtime does not necessarily contain AMD’s complete Ryzen AI EP packaging.

Build the application using **only the headers and libraries belonging to the AMD ONNX Runtime installation that contains the real EP**.

For example:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${RYZEN_AI_INSTALLATION_PATH:?Activate/install Ryzen AI first}"

ORT_ROOT="$RYZEN_AI_INSTALLATION_PATH/onnxruntime"
ORT_INCLUDE="$ORT_ROOT/include"
ORT_LIB="$ORT_ROOT/lib"

test -f "$ORT_INCLUDE/onnxruntime_cxx_api.h"
test -f "$ORT_LIB/libonnxruntime.so"
test -e "$ORT_LIB/libonnxruntime_vitisai_ep.so"

g++ -std=c++20 -O3 -march=native -flto \
    server_npu.cpp -o puck_eye_npu_server \
    -I/usr/include/uWebSockets \
    -I"$ORT_INCLUDE" \
    -L"$ORT_LIB" \
    -Wl,-rpath,"$ORT_LIB" \
    -Wl,-rpath,'$ORIGIN/lib' \
    $(pkg-config --cflags --libs libturbojpeg) \
    -lusockets -lonnxruntime -lz -lpthread

echo "[SUCCESS] Built against $ORT_LIB"
ldd ./puck_eye_npu_server | grep onnxruntime
```

If AMD’s package puts headers or libraries elsewhere, derive both paths from that same package. Do not silently fall back to `/usr/include/onnxruntime`.

Also remove OpenCV from the build because your code does not use it:

```bash
$(pkg-config --cflags --libs libturbojpeg)
```

rather than:

```bash
$(pkg-config --cflags --libs libturbojpeg opencv4)
```

---

# 5. Do not use the custom ORT build unless it actually built the EP

Inspect your custom build:

```bash
find /home/stack/repos/onnxruntime-build/Release -maxdepth 1 \
    -name 'libonnxruntime*.so*' -ls
```

If it contains only:

```text
libonnxruntime.so
libonnxruntime_providers_shared.so
libonnxruntime_providers_vitisai.so
```

but lacks:

```text
libonnxruntime_vitisai_ep.so
```

then that build is incomplete for the ORT configuration you are using.

The clean options are:

### Recommended

Install AMD’s complete Ryzen AI for Linux package and build against its bundled ONNX Runtime.

### Alternative

Rebuild the exact AMD-supported ONNX Runtime revision using AMD’s documented build procedure and its matching Vitis AI/Ryzen AI sources.

Do not use arbitrary upstream ONNX Runtime `main` plus libraries copied from a Ryzen AI SDK. EP/Core ABI compatibility is not guaranteed.

Check source/runtime versions:

```bash
strings /home/stack/repos/onnxruntime-build/Release/libonnxruntime.so |
    grep -m1 -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true

git -C /home/stack/repos/onnxruntime-src rev-parse HEAD
git -C /home/stack/repos/onnxruntime-src status --short
```

---

# 6. Use the supported Linux baseline before debugging Fedora-specific failures

AMD’s current Linux instructions specify:

- Ubuntu 24.04 LTS
- kernel 6.10 or newer
- supported STX/KRK platform
- the official Ryzen AI Linux package and corresponding NPU driver ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/linux.html?utm_source=openai))

The HX 370 is the correct hardware class, but Fedora 44 remains outside the documented distribution baseline.

For a high-confidence diagnosis:

1. Boot or install Ubuntu 24.04.
2. Install the exact AMD Ryzen AI Linux package and matching driver.
3. Run AMD’s included quick test/example.
4. Run a minimal ONNX Runtime NPU test.
5. Only then build the WebSocket server.

A container may help with userspace dependencies, but it does not replace a compatible host kernel driver, device access, firmware, or ioctl ABI.

If the same official package works on Ubuntu but not Fedora, you have isolated the problem to the unsupported host environment rather than your model or C++ code.

---

# 7. Verify that Linux sees the XDNA NPU

Start with PCI enumeration:

```bash
lspci -nnk | grep -A4 -Ei '1022:17f0|NPU|XDNA|AMD'
```

For Strix Point, AMD documentation identifies the NPU PCI device family with AMD vendor `0x1022` and device `0x17F0`. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/1.5/app_development.html?utm_source=openai))

Check loaded modules:

```bash
lsmod | grep -Ei 'amdxdna|xrt|zocl'
```

Check kernel logs:

```bash
sudo dmesg --color=always |
    grep -Ei 'amdxdna|xdna|xrt|npu|firmware|iommu|error|fail'
```

Check device nodes:

```bash
find /dev -maxdepth 2 \
    \( -iname '*accel*' -o -iname '*xdna*' -o -iname '*dri*' \) \
    -ls
```

Check permissions:

```bash
ls -la /dev/accel 2>/dev/null || true
ls -la /dev/accel/* 2>/dev/null || true
id
```

Check XRT:

```bash
xrt-smi examine
xrt-smi validate 2>/dev/null || true
```

You need all of the following:

- PCI NPU device present
- correct driver bound
- device node created
- user has permission
- XRT can enumerate the device
- driver/runtime versions are compatible

If `xrt-smi examine` cannot see the NPU, stop debugging ONNX Runtime. Fix the driver/device layer first.

---

# 8. Test the AMD installation independently of your application

Before running your server, test provider registration using the Python ONNX Runtime shipped with the same Ryzen AI installation:

```bash
python3 - <<'PY'
import onnxruntime as ort

print("ORT version:", ort.__version__)
print("ORT module:", ort.__file__)
print("Available providers:", ort.get_available_providers())

assert "VitisAIExecutionProvider" in ort.get_available_providers()
PY
```

Important: `ort.__file__` must point into the AMD Ryzen AI environment, not Fedora’s system Python packages.

If it prints only something like:

```text
['CPUExecutionProvider']
```

then the AMD runtime is not active or is incomplete.

Test loading the model:

```bash
python3 - <<'PY'
import os
import onnxruntime as ort

model = os.path.abspath("puck-eye-seg-s_bf16_ctx.onnx")
config = os.path.abspath("vaip_bf16_config.json")
cache = os.path.abspath("npu_cache_test")

print("ORT:", ort.__file__)
print("Providers:", ort.get_available_providers())
print("Model:", model)
print("Config:", config)

assert os.path.isfile(model), model
assert os.path.isfile(config), config
assert "VitisAIExecutionProvider" in ort.get_available_providers()

options = {
    "config_file": config,
    "cacheDir": cache,
    "cacheKey": "puck_eye_test",
}

session = ort.InferenceSession(
    model,
    providers=["VitisAIExecutionProvider"],
    provider_options=[options],
)

print("Session providers:", session.get_providers())
print("Inputs:", [(x.name, x.shape, x.type) for x in session.get_inputs()])
print("Outputs:", [(x.name, x.shape, x.type) for x in session.get_outputs()])
PY
```

AMD documents this Vitis AI provider-based loading flow and compilation/cache mechanism. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/modelrun.html?utm_source=openai))

If Python fails with the same missing `.so`, the installation is broken independently of your C++ code.

---

# 9. Use a matching AMD configuration file

This is currently hard-coded:

```cpp
{"config_file", "vaip_bf16_config.json"}
```

The configuration file must come from the **same Ryzen AI release as the Vitis AI EP**. AMD explicitly warns against mixing a configuration file from one release with an EP from another. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/1.3/app_development.html?utm_source=openai))

Validate it before starting:

```bash
CONFIG="$(readlink -f vaip_bf16_config.json)"
test -f "$CONFIG" || {
    echo "Missing config: $CONFIG"
    exit 1
}
```

Prefer passing an absolute path:

```cpp
std::unordered_map<std::string, std::string> vitis_opts = {
    {"config_file", "/absolute/path/to/vaip_bf16_config.json"},
    {"cacheDir", "/absolute/path/to/npu_cache_bf16"},
    {"cacheKey", "puck-eye-seg-s-bf16"}
};
```

Provider option spelling can vary by release. AMD examples have used camel-case cache keys such as `cacheDir`/`cacheKey`; other documentation and examples may show snake-case forms. Use the options documented or included with your exact installed release rather than guessing. ([github.com](https://github.com/amd/RyzenAI-SW/blob/main/Demos/ASR/Whisper/README.md?utm_source=openai))

Clear caches whenever changing:

- Ryzen AI version
- Vitis AI EP
- NPU driver
- model
- configuration file

```bash
rm -rf ./npu_cache_bf16 ./npu_cache_test
```

AMD says caches should not be reused across EP or driver versions. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/1.3/app_development.html?utm_source=openai))

---

# 10. Disable CPU fallback while debugging

Your current code hides NPU initialization problems:

```cpp
catch (...) {
    // create CPU session
}
```

If NPU execution is mandatory, fail closed:

```cpp
try {
    session_opts.AppendExecutionProvider(
        "VitisAIExecutionProvider", vitis_opts);

    session = std::make_unique<Ort::Session>(
        env, compiled_model_path.c_str(), session_opts);

    std::cout << "[INFO] Vitis AI session created.\n";
} catch (const Ort::Exception& e) {
    std::cerr << "[FATAL] Vitis AI EP initialization failed: "
              << e.what() << '\n';
    throw;
}
```

Also remove the unused variable:

```cpp
bool ep_loaded = false;
```

This prevents a production run from appearing healthy while using the CPU.

However, be precise about “NPU only”: Vitis AI can partition an ONNX graph, run supported subgraphs on the NPU, and leave unsupported subgraphs on the CPU. A successfully loaded Vitis AI session does not prove that every operator ran on the NPU. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/modelrun.html?utm_source=openai))

Your JPEG decode, resize/normalization, NMS, and mask generation are explicitly CPU code regardless.

---

# 11. Enable verbose ORT logging and profiling

Change:

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "BF16_NPU_Server"};
```

temporarily to:

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "BF16_NPU_Server"};
```

Add:

```cpp
session_opts.SetLogSeverityLevel(0);
session_opts.SetLogVerbosityLevel(1);
session_opts.EnableProfiling("ort_npu_profile");
```

After inference:

```cpp
Ort::AllocatorWithDefaultOptions allocator;
auto profile = session->EndProfilingAllocated(allocator);
std::cout << "[PROFILE] " << profile.get() << '\n';
```

Inspect the profile and provider logs for node assignment. You are looking for:

- Vitis AI compilation messages
- generated cache/context artifacts
- nodes assigned to `VitisAIExecutionProvider`
- unexpected CPU nodes
- unsupported-operator warnings

A nonempty `.rai` or EP-context/cache artifact is stronger evidence that NPU compilation occurred than simply seeing the provider name. AMD documents the compilation and cache/context workflow. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/modelrun.html?utm_source=openai))

---

# 12. BF16-specific issue you may encounter next

Once the loader is fixed, BF16 deployment may impose an additional constraint.

AMD documents that BF16 is supported on STX/KRK or newer devices, but some deployment configurations require precompiled BF16 artifacts rather than on-the-fly C++ compilation. Current AMD guidance recommends precompiled models/context caches for deployment. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/modelrun.html?utm_source=openai))

Therefore verify what this file actually is:

```text
puck-eye-seg-s_bf16_ctx.onnx
```

Check it:

```bash
python3 - <<'PY'
import onnx

p = "puck-eye-seg-s_bf16_ctx.onnx"
m = onnx.load(p)

print("IR version:", m.ir_version)
print("Opsets:", [(x.domain, x.version) for x in m.opset_import])
print("Inputs:", [x.name for x in m.graph.input])
print("Outputs:", [x.name for x in m.graph.output])

for p in m.metadata_props:
    print("Metadata:", p.key, "=", p.value)
PY
```

Do not assume that a filename ending in `_ctx.onnx` makes it a valid EP context model. It must have been generated by the compatible AMD compiler/runtime for the correct hardware target and release.

AMD currently recommends ONNX opset 17 and publishes a BF16 operator-support matrix. ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/modelrun.html?utm_source=openai))

---

# 13. Corrected runtime preflight script

A stricter version should look approximately like this:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-puck-eye-seg-s_bf16_ctx.onnx}"
PORT="${2:-18888}"

: "${RYZEN_AI_INSTALLATION_PATH:?RYZEN_AI_INSTALLATION_PATH is not set}"

ORT_ROOT="${RYZEN_AI_INSTALLATION_PATH}/onnxruntime"

if [ -d "$ORT_ROOT/lib" ]; then
    AMD_ORT_LIB="$ORT_ROOT/lib"
elif [ -d "$ORT_ROOT/capi" ]; then
    AMD_ORT_LIB="$ORT_ROOT/capi"
else
    echo "[FATAL] Cannot locate AMD ONNX Runtime under $ORT_ROOT"
    exit 1
fi

ORT_SO="$AMD_ORT_LIB/libonnxruntime.so"
EP_SO="$AMD_ORT_LIB/libonnxruntime_vitisai_ep.so"

test -e "$ORT_SO" || {
    echo "[FATAL] Missing $ORT_SO"
    exit 1
}

test -e "$EP_SO" || {
    echo "[FATAL] Missing the real Vitis AI EP library:"
    echo "  $EP_SO"
    echo "Do not substitute libonnxruntime_providers_vitisai.so."
    exit 1
}

EP_REAL="$(readlink -e "$EP_SO")"
test -n "$EP_REAL" || {
    echo "[FATAL] Broken EP symlink: $EP_SO"
    exit 1
}

for lib in "$ORT_SO" "$EP_REAL"; do
    if ldd "$lib" | grep -q 'not found'; then
        echo "[FATAL] Missing dependencies for $lib:"
        ldd "$lib" | grep 'not found'
        exit 1
    fi
done

test -f "$MODEL_PATH" || {
    echo "[FATAL] Model not found: $MODEL_PATH"
    exit 1
}

test -f vaip_bf16_config.json || {
    echo "[FATAL] Missing vaip_bf16_config.json"
    exit 1
}

export LD_LIBRARY_PATH="$AMD_ORT_LIB:/opt/xilinx/xrt/lib:/opt/xilinx/xrt/lib64:${LD_LIBRARY_PATH:-}"

echo "[INFO] Executable dependencies:"
ldd ./puck_eye_npu_server | grep -E 'onnxruntime|vitis|ryzen|xrt' || true

if command -v xrt-smi >/dev/null 2>&1; then
    xrt-smi examine
else
    echo "[FATAL] xrt-smi is unavailable"
    exit 1
fi

rm -rf ./npu_cache_bf16

exec ./puck_eye_npu_server "$MODEL_PATH" "$PORT"
```

Do not add `/lib/x86_64-linux-gnu` to the front of `LD_LIBRARY_PATH` on Fedora. That is a Debian/Ubuntu path and may introduce further runtime confusion.

---

# Recommended resolution order

1. **Remove the fake `providers_vitisai → vitisai_ep` symlink.**
2. Confirm whether the real `libonnxruntime_vitisai_ep.so` exists.
3. If absent, install AMD’s complete Ryzen AI Linux package; do not fabricate the file.
4. Use Ubuntu 24.04 as the supported reference environment.
5. Verify NPU visibility with PCI, driver logs, device nodes, and `xrt-smi`.
6. Run AMD’s supplied quick test.
7. Run the minimal Python `VitisAIExecutionProvider` test.
8. Build your application against the exact headers and libraries from that same AMD installation.
9. Use the matching config file and clear the cache.
10. Remove CPU fallback and enable verbose profiling.
11. Validate that the model produces NPU cache/context artifacts and inspect node placement.
12. Only then debug model support, BF16 compilation, output layouts, or performance.

The most likely immediate fix is: **replace `/home/stack/repos/onnxruntime-build/Release` with the library directory from a complete, internally consistent AMD Ryzen AI installation containing the real `libonnxruntime_vitisai_ep.so`, then rebuild and run against that same installation.**

## References

1. [Application Development — Ryzen AI Software 1.3 documentation](https://ryzenai.docs.amd.com/en/1.3/app_development.html?utm_source=openai)
2. [Linux Installation Instructions — Ryzen AI Software 1.7.1 documentation](https://ryzenai.docs.amd.com/en/latest/linux.html?utm_source=openai)
3. [Application Development — Ryzen AI Software 1.5 documentation](https://ryzenai.docs.amd.com/en/1.5/app_development.html?utm_source=openai)
4. [Model Compilation and Deployment — Ryzen AI Software 1.7.1 documentation](https://ryzenai.docs.amd.com/en/latest/modelrun.html?utm_source=openai)
5. [RyzenAI-SW/Demos/ASR/Whisper/README.md at main · amd/RyzenAI-SW · GitHub](https://github.com/amd/RyzenAI-SW/blob/main/Demos/ASR/Whisper/README.md?utm_source=openai)
