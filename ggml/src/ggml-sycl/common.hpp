//
// SYCL backend device runtime layer.
//
// Mirrors the role of ggml-cuda/common.cuh: owns the device list, the per-device SYCL queue
// (the SYCL analogue of a CUDA stream), and the small set of helpers every op file includes.
// A ggml "device" maps 1:1 to a sycl::device discovered via the level_zero gpu platform; a
// ggml "stream" maps 1:1 to a sycl::queue on that device (in-order, profiling enabled so
// per-kernel timing/events work for the profiler).
//
#pragma once

#include <sycl/sycl.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-impl.h"

#define GGML_SYCL_MAX_DEVICES 48

// Fatal error helper: SYCL reports device errors via exceptions, not return codes, so every
// queue-affecting call site is wrapped and asserts on failure (mirrors CUDA_CHECK in ggml-cuda).
#define SYCL_CHECK(err)                                                                   \
    do {                                                                                  \
        try {                                                                             \
            (err);                                                                        \
        } catch (sycl::exception const & e) {                                            \
            GGML_LOG_ERROR("%s: SYCL error at %s:%d: %s\n", __func__, __FILE__, __LINE__, \
                            e.what());                                                    \
            GGML_ABORT("SYCL error");                                                     \
        }                                                                                 \
    } while (0)

struct ggml_sycl_device_info {
    sycl::device dev;
    std::string   name;
    std::string   description;
    size_t        total_vram = 0;
    int           max_work_group_size = 0;
};

// One process-wide device/queue registry. Devices are enumerated once at first use; each device
// owns a single in-order queue used for both compute and copies (no multi-stream overlap yet --
// see integration/index.json "memory"/"runtime" surfaces for the residual work item).
struct ggml_sycl_device_mgr {
    std::vector<ggml_sycl_device_info> infos;
    std::vector<sycl::queue>           queues;

    static ggml_sycl_device_mgr & instance() {
        static ggml_sycl_device_mgr mgr;
        return mgr;
    }

    ggml_sycl_device_mgr() {
        try {
            auto platforms = sycl::platform::get_platforms();
            for (auto & plat : platforms) {
                // Prefer the Level Zero backend (matches ONEAPI_DEVICE_SELECTOR=level_zero:gpu
                // used elsewhere in this project's runner config).
                for (auto & dev : plat.get_devices(sycl::info::device_type::gpu)) {
                    if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) {
                        continue;
                    }
                    ggml_sycl_device_info info;
                    info.dev                 = dev;
                    info.name                = dev.get_info<sycl::info::device::name>();
                    info.description         = info.name;
                    info.total_vram          = dev.get_info<sycl::info::device::global_mem_size>();
                    info.max_work_group_size = (int) dev.get_info<sycl::info::device::max_work_group_size>();
                    infos.push_back(info);
                }
            }
        } catch (sycl::exception const & e) {
            GGML_LOG_ERROR("%s: failed to enumerate SYCL devices: %s\n", __func__, e.what());
        }

        queues.reserve(infos.size());
        for (auto & info : infos) {
            queues.emplace_back(info.dev, sycl::property_list{ sycl::property::queue::in_order() });
        }
    }

    int device_count() const { return (int) infos.size(); }

    sycl::queue & queue(int device) {
        GGML_ASSERT(device >= 0 && device < (int) queues.size());
        return queues[device];
    }
};

static inline sycl::queue & ggml_sycl_queue(int device) {
    return ggml_sycl_device_mgr::instance().queue(device);
}

static inline int ggml_sycl_device_count() {
    return ggml_sycl_device_mgr::instance().device_count();
}

// Per-backend-instance context (the SYCL analogue of the CUDA backend context): which physical
// device this ggml_backend targets.
struct ggml_backend_sycl_context {
    int device;

    explicit ggml_backend_sycl_context(int device) : device(device) {}

    sycl::queue & stream() { return ggml_sycl_queue(device); }
};
