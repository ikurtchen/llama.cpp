//
// SYCL backend for ggml. Backend switch: -DGGML_SYCL=ON (see ggml/CMakeLists.txt).
// Mirrors the structure of ggml-cuda/ggml-cuda.cu (backend/device/reg + buffer type + dispatch),
// hand-written against SYCL 2020 / USM instead of the CUDA runtime API.
//
#include "ggml-sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "common.hpp"
#include "cpy.hpp"
#include "scale.hpp"
#include "clamp.hpp"
#include "unary.hpp"
#include "rope.hpp"
#include "diagmask.hpp"
#include "concat.hpp"
#include "arange.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

// ------------------------------------------------------------------------------------------------
// Buffer type / buffer: device memory via SYCL USM (device-only allocations), the USM analogue of
// cudaMalloc/cudaFree/cudaMemcpy used by ggml-cuda's buffer type.
// ------------------------------------------------------------------------------------------------

struct ggml_backend_sycl_buffer_type_context {
    int         device;
    std::string name;
};

struct ggml_backend_sycl_buffer_context {
    int    device;
    void * dev_ptr = nullptr;
};

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer);

static const char * ggml_backend_sycl_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    return ggml_sycl_device_mgr::instance().infos[ctx->device].name.c_str();
}

static void ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl::free(ctx->dev_ptr, ggml_sycl_queue(ctx->device));
    delete ctx;
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                     uint8_t value, size_t offset, size_t size) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl::queue & q = ggml_sycl_queue(ctx->device);
    SYCL_CHECK(q.memset((char *) tensor->data + offset, value, size).wait());
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                  const void * data, size_t offset, size_t size) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl::queue & q = ggml_sycl_queue(ctx->device);
    SYCL_CHECK(q.memcpy((char *) tensor->data + offset, data, size).wait());
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                  void * data, size_t offset, size_t size) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl::queue & q = ggml_sycl_queue(ctx->device);
    SYCL_CHECK(q.memcpy(data, (const char *) tensor->data + offset, size).wait());
}

static bool ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    if (!ggml_backend_buffer_is_sycl(src->buffer)) {
        return false;
    }
    sycl::queue & q = ggml_sycl_queue(ctx->device);
    SYCL_CHECK(q.memcpy(dst->data, src->data, ggml_nbytes(src)).wait());
    return true;
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl::queue & q = ggml_sycl_queue(ctx->device);
    SYCL_CHECK(q.memset(ctx->dev_ptr, value, buffer->size).wait());
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ nullptr,
};

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_sycl_buffer_free_buffer;
}

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;
    sycl::queue & q = ggml_sycl_queue(buft_ctx->device);

    // USM device allocation -- the SYCL analogue of cudaMalloc. Round up to avoid 0-byte allocs.
    void * dev_ptr = sycl::malloc_device(std::max(size, (size_t) 1), q);
    if (!dev_ptr) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes on SYCL device %d\n", __func__, size, buft_ctx->device);
        return nullptr;
    }

    ggml_backend_sycl_buffer_context * ctx = new ggml_backend_sycl_buffer_context{ buft_ctx->device, dev_ptr };
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mtx;
    static std::vector<ggml_backend_buffer_type> buft_list;
    std::lock_guard<std::mutex> lock(mtx);

    if (buft_list.empty()) {
        int n = ggml_sycl_device_count();
        buft_list.resize(n);
        for (int i = 0; i < n; i++) {
            auto * ctx = new ggml_backend_sycl_buffer_type_context{ i, std::string("SYCL") + std::to_string(i) };
            buft_list[i] = ggml_backend_buffer_type{ ggml_backend_sycl_buffer_type_interface, nullptr, ctx };
        }
    }
    GGML_ASSERT(device >= 0 && device < (int) buft_list.size());
    return &buft_list[device];
}

// Multi-GPU tensor-parallel split buffer / comm are out of scope for this migration (single-GPU
// B70 target) -- waived explicitly in .sycl/state/integration/readiness.json rather than stubbed
// silently.
ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    GGML_UNUSED(tensor_split);
    GGML_LOG_ERROR("%s: multi-GPU split buffers are not implemented in the SYCL backend (waived, single-GPU target)\n", __func__);
    return nullptr;
}

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) {
    GGML_UNUSED(backends); GGML_UNUSED(n_backends);
    return nullptr;
}
void ggml_backend_sycl_comm_free(void * comm_ctx) { GGML_UNUSED(comm_ctx); }
bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
    GGML_UNUSED(comm_ctx); GGML_UNUSED(tensors);
    return false;
}

// Host (pinned) buffer: falls back to the plain CPU buffer type for now (correct, just without
// the transfer-speed benefit of USM host-pinned memory) -- see integration/index.json "memory".
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void) {
    return ggml_backend_cpu_buffer_type();
}

// ------------------------------------------------------------------------------------------------
// Backend
// ------------------------------------------------------------------------------------------------

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev);

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {
    return ggml_backend_sycl_device_get_name(backend->device);
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * ctx = (ggml_backend_sycl_context *) backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {
    ggml_backend_sycl_context * ctx = (ggml_backend_sycl_context *) backend->context;
    ctx->stream().wait();
}

// Central op dispatch. Each case is implemented in its own <op>.cpp, mirroring ggml-cuda's file
// layout; this switch is extended one kernel at a time as `migrate` proceeds (see
// .sycl/state/kernels/index.json for the full worklist and status).
static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            ggml_sycl_cpy(ctx, dst->src[0], dst);
            return true;
        case GGML_OP_SCALE:
            ggml_sycl_op_scale(ctx, dst);
            return true;
        case GGML_OP_CLAMP:
            ggml_sycl_op_clamp(ctx, dst);
            return true;
        case GGML_OP_UNARY:
            ggml_sycl_op_unary(ctx, dst);
            return true;
        case GGML_OP_GLU:
            ggml_sycl_op_glu(ctx, dst);
            return true;
        case GGML_OP_SQR:
            ggml_sycl_op_sqr(ctx, dst);
            return true;
        case GGML_OP_SQRT:
            ggml_sycl_op_sqrt(ctx, dst);
            return true;
        case GGML_OP_SIN:
            ggml_sycl_op_sin(ctx, dst);
            return true;
        case GGML_OP_COS:
            ggml_sycl_op_cos(ctx, dst);
            return true;
        case GGML_OP_LOG:
            ggml_sycl_op_log(ctx, dst);
            return true;
        case GGML_OP_ROPE:
            ggml_sycl_op_rope(ctx, dst);
            return true;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_op_diag_mask_inf(ctx, dst);
            return true;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            return true;
        case GGML_OP_ARANGE:
            ggml_sycl_op_arange(ctx, dst);
            return true;
        default:
            return false;
    }
}

static enum ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_sycl_context * ctx = (ggml_backend_sycl_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        bool ok = ggml_sycl_compute_forward(*ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x62, 0x4c, 0x51, 0xa3, 0x9e, 0x1d, 0x4a, 0x7b, 0x9c, 0x2e, 0x8f, 0x5a, 0x3d, 0x71, 0x0e, 0x44 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    if (device < 0 || device >= ggml_sycl_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }
    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    return new ggml_backend{
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx,
    };
}

// ------------------------------------------------------------------------------------------------
// Device
// ------------------------------------------------------------------------------------------------

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    int device = (int) (intptr_t) dev->context;
    // "SYCL0", "SYCL1", ... -- mirrors ggml-cuda's "CUDA0"/"CUDA1" naming, which is what
    // test-backend-ops' `-b <name>` filter and the project's own tooling match against.
    static std::vector<std::string> names;
    if ((int) names.size() <= device) {
        names.resize(device + 1);
    }
    if (names[device].empty()) {
        names[device] = std::string("SYCL") + std::to_string(device);
    }
    return names[device].c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    int device = (int) (intptr_t) dev->context;
    return ggml_sycl_device_mgr::instance().infos[device].description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    int device = (int) (intptr_t) dev->context;
    auto & info = ggml_sycl_device_mgr::instance().infos[device];
    *total = info.total_vram;
    *free  = info.total_vram; // SYCL has no direct "free VRAM" query; report capacity (see lessons.md)
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    int device = (int) (intptr_t) dev->context;
    return ggml_backend_sycl_init(device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    int device = (int) (intptr_t) dev->context;
    return ggml_backend_sycl_buffer_type(device);
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_sycl_supports_cpy(op);
        case GGML_OP_SCALE:
            return ggml_sycl_supports_scale(op);
        case GGML_OP_CLAMP:
            return ggml_sycl_supports_clamp(op);
        case GGML_OP_UNARY:
            return ggml_sycl_supports_unary(op);
        case GGML_OP_GLU:
            return ggml_sycl_supports_glu(op);
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_sycl_supports_unary_f32f16(op);
        case GGML_OP_ROPE:
            return ggml_sycl_supports_rope(op);
        case GGML_OP_DIAG_MASK_INF:
            return ggml_sycl_supports_diag_mask_inf(op);
        case GGML_OP_CONCAT:
            return ggml_sycl_supports_concat(op);
        case GGML_OP_ARANGE:
            return ggml_sycl_supports_arange(op);
        default:
            return false;
    }
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;
    int device = (int) (intptr_t) dev->context;
    return buft_ctx->device == device;
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name             = */ ggml_backend_sycl_device_get_name,
    /* .get_description      = */ ggml_backend_sycl_device_get_description,
    /* .get_memory           = */ ggml_backend_sycl_device_get_memory,
    /* .get_type             = */ ggml_backend_sycl_device_get_type,
    /* .get_props            = */ ggml_backend_sycl_device_get_props,
    /* .init_backend         = */ ggml_backend_sycl_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ------------------------------------------------------------------------------------------------
// Backend registry
// ------------------------------------------------------------------------------------------------

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_device> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static void * ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name         = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device       = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address = */ ggml_backend_sycl_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_sycl_reg(void) {
    static ggml_backend_reg reg;
    static ggml_backend_sycl_reg_context * ctx = nullptr;
    static std::once_flag once;

    std::call_once(once, [&]() {
        ctx = new ggml_backend_sycl_reg_context();
        int n = ggml_sycl_device_count();
        ctx->devices.resize(n);
        for (int i = 0; i < n; i++) {
            ctx->devices[i] = ggml_backend_device{
                /* .iface   = */ ggml_backend_sycl_device_interface,
                /* .reg     = */ &reg,
                /* .context = */ (void *) (intptr_t) i,
            };
        }
        reg = ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_sycl_reg_interface,
            /* .context     = */ ctx,
        };
    });

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)

// ------------------------------------------------------------------------------------------------
// Misc introspection API declared in ggml-sycl.h
// ------------------------------------------------------------------------------------------------

void ggml_backend_sycl_print_sycl_devices(void) {
    auto & mgr = ggml_sycl_device_mgr::instance();
    for (int i = 0; i < mgr.device_count(); i++) {
        printf("  Device %d: %s\n", i, mgr.infos[i].name.c_str());
    }
}

void ggml_backend_sycl_get_gpu_list(int * id_list, int max_len) {
    int n = std::min(ggml_sycl_device_count(), max_len);
    for (int i = 0; i < n; i++) {
        id_list[i] = i;
    }
    for (int i = n; i < max_len; i++) {
        id_list[i] = -1;
    }
}

void ggml_backend_sycl_get_device_description(int device, char * description, size_t description_size) {
    auto & mgr = ggml_sycl_device_mgr::instance();
    if (device < 0 || device >= mgr.device_count()) {
        description[0] = '\0';
        return;
    }
    snprintf(description, description_size, "%s", mgr.infos[device].description.c_str());
}

int ggml_backend_sycl_get_device_count() {
    return ggml_sycl_device_count();
}

void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total) {
    auto & mgr = ggml_sycl_device_mgr::instance();
    if (device < 0 || device >= mgr.device_count()) {
        *free = 0; *total = 0;
        return;
    }
    *total = mgr.infos[device].total_vram;
    *free  = mgr.infos[device].total_vram;
}
