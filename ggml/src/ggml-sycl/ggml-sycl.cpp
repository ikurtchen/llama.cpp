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
#include "binbcast.hpp"
#include "unary.hpp"
#include "softmax.hpp"
#include "rope.hpp"
#include "diagmask.hpp"
#include "concat.hpp"
#include "arange.hpp"
#include "argsort.hpp"
#include "argmax.hpp"
#include "fattn.hpp"
#include "norm.hpp"
#include "top-k.hpp"
#include "getrows.hpp"
#include "set-rows.hpp"
#include "set.hpp"
#include "add-id.hpp"
#include "diag.hpp"
#include "pad.hpp"
#include "pad_reflect_1d.hpp"
#include "roll.hpp"
#include "tsembd.hpp"
#include "tri.hpp"
#include "opt-step-sgd.hpp"
#include "fill.hpp"
#include "snake.hpp"
#include "softcap.hpp"
#include "im2col.hpp"
#include "conv2d.hpp"
#include "conv2d-dw.hpp"
#include "conv2d-transpose.hpp"
#include "conv-transpose-1d.hpp"
#include "col2im-1d.hpp"
#include "pool1d.hpp"
#include "pool2d.hpp"
#include "upscale.hpp"
#include "conv3d.hpp"
#include "reduce_rows.hpp"
#include "acc.hpp"
#include "mmf.hpp"
#include "fwht.hpp"
#include "opt-step-adamw.hpp"
#include "cross-entropy-loss.hpp"
#include "cumsum.hpp"
#include "out-prod.hpp"
#include "solve-tri.hpp"
#include "dsv4-hc.hpp"
#include "lightning-indexer.hpp"
#include "gla.hpp"
#include "ssm-conv.hpp"
#include "wkv.hpp"
#include "gated_delta_net.hpp"
#include "ssm-scan.hpp"

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
        case GGML_OP_REPEAT:
            ggml_sycl_op_repeat(ctx, dst);
            return true;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_op_repeat_back(ctx, dst);
            return true;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
            ggml_sycl_op_add(ctx, dst);
            return true;
        case GGML_OP_SUB:
            ggml_sycl_op_sub(ctx, dst);
            return true;
        case GGML_OP_MUL:
            ggml_sycl_op_mul(ctx, dst);
            return true;
        case GGML_OP_DIV:
            ggml_sycl_op_div(ctx, dst);
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
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            return true;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            return true;
        case GGML_OP_NORM:
            ggml_sycl_op_norm(ctx, dst);
            return true;
        case GGML_OP_RMS_NORM:
            ggml_sycl_op_rms_norm(ctx, dst);
            return true;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_op_group_norm(ctx, dst);
            return true;
        case GGML_OP_L2_NORM:
            ggml_sycl_op_l2_norm(ctx, dst);
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
        case GGML_OP_ARGSORT:
            ggml_sycl_op_argsort(ctx, dst);
            return true;
        case GGML_OP_ARGMAX:
            ggml_sycl_op_argmax(ctx, dst);
            return true;
        case GGML_OP_MUL_MAT:
            ggml_sycl_op_mul_mat(ctx, dst);
            return true;
        case GGML_OP_OUT_PROD:
            ggml_sycl_out_prod(ctx, dst);
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_op_flash_attn_ext(ctx, dst);
            return true;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            return true;
        case GGML_OP_IM2COL:
            ggml_sycl_op_im2col(ctx, dst);
            return true;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_op_im2col_3d(ctx, dst);
            return true;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            return true;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            return true;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            return true;
        case GGML_OP_COL2IM_1D:
            ggml_sycl_op_col2im_1d(ctx, dst);
            return true;
        case GGML_OP_POOL_1D:
            ggml_sycl_op_pool1d(ctx, dst);
            return true;
        case GGML_OP_POOL_2D:
            ggml_sycl_op_pool2d(ctx, dst);
            return true;
        case GGML_OP_UPSCALE:
            ggml_sycl_op_upscale(ctx, dst);
            return true;
        case GGML_OP_CONV_3D:
            ggml_sycl_op_conv3d(ctx, dst);
            return true;
        case GGML_OP_GET_ROWS:
            ggml_sycl_op_get_rows(ctx, dst);
            return true;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            return true;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            return true;
        case GGML_OP_ADD_ID:
            ggml_sycl_op_add_id(ctx, dst);
            return true;
        case GGML_OP_DIAG:
            ggml_sycl_op_diag(ctx, dst);
            return true;
        case GGML_OP_PAD:
            ggml_sycl_op_pad(ctx, dst);
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx, dst);
            return true;
        case GGML_OP_ROLL:
            ggml_sycl_op_roll(ctx, dst);
            return true;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            return true;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            return true;
        case GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            return true;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            return true;
        case GGML_OP_FILL:
            ggml_sycl_op_fill(ctx, dst);
            return true;
        case GGML_OP_SUM:
            ggml_sycl_op_sum(ctx, dst);
            return true;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_op_sum_rows(ctx, dst);
            return true;
        case GGML_OP_MEAN:
            ggml_sycl_op_mean(ctx, dst);
            return true;
        case GGML_OP_ACC:
            ggml_sycl_op_acc(ctx, dst);
            return true;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            return true;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            return true;
        case GGML_OP_CUMSUM:
            ggml_sycl_op_cumsum(ctx, dst);
            return true;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_op_solve_tri(ctx, dst);
            return true;
        case GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            return true;
        case GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            return true;
        case GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            return true;
        case GGML_OP_SSM_CONV:
            ggml_sycl_op_ssm_conv(ctx, dst);
            return true;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_op_ssm_scan(ctx, dst);
            return true;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            return true;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            return true;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            return true;
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_op_gated_delta_net(ctx, dst);
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
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

        if (node->op == GGML_OP_MUL && ggml_sycl_can_fuse_snake(cgraph, i)) {
            const ggml_tensor * mul0 = cgraph->nodes[i + 0];
            const ggml_tensor * sqr  = cgraph->nodes[i + 2];
            const ggml_tensor * mul1 = cgraph->nodes[i + 3];
            ggml_tensor *       add  = cgraph->nodes[i + 4];

            const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
            const ggml_tensor * a = x == mul0->src[0] ? mul0->src[1] : mul0->src[0];
            const ggml_tensor * inv_b = mul1->src[0] == sqr ? mul1->src[1] : mul1->src[0];

            ggml_sycl_op_snake_fused(*ctx, x, a, inv_b, add);
            i += 4;
            continue;
        }

        if (node->op == GGML_OP_SCALE && ggml_sycl_can_fuse_softcap(cgraph, i)) {
            ggml_sycl_op_softcap(*ctx, cgraph->nodes[i + 2], node);
            i += 2;
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
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return ggml_sycl_supports_binbcast(op);
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
        case GGML_OP_SOFT_MAX:
            return ggml_sycl_supports_soft_max(op);
        case GGML_OP_SOFT_MAX_BACK:
            return ggml_sycl_supports_soft_max_back(op);
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return ggml_sycl_supports_norm(op);
        case GGML_OP_ROPE:
            return ggml_sycl_supports_rope(op);
        case GGML_OP_DIAG_MASK_INF:
            return ggml_sycl_supports_diag_mask_inf(op);
        case GGML_OP_CONCAT:
            return ggml_sycl_supports_concat(op);
        case GGML_OP_ARANGE:
            return ggml_sycl_supports_arange(op);
        case GGML_OP_ARGSORT:
            return ggml_sycl_supports_argsort(op);
        case GGML_OP_ARGMAX:
            return ggml_sycl_supports_argmax(op);
        case GGML_OP_MUL_MAT:
            return ggml_sycl_supports_mul_mat(op);
        case GGML_OP_OUT_PROD:
            return ggml_sycl_supports_out_prod(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_supports_flash_attn_ext(op);
        case GGML_OP_TOP_K:
            return ggml_sycl_supports_top_k(op);
        case GGML_OP_IM2COL:
            return ggml_sycl_supports_im2col(op);
        case GGML_OP_IM2COL_3D:
            return ggml_sycl_supports_im2col_3d(op);
        case GGML_OP_CONV_2D:
            return ggml_sycl_supports_conv2d(op);
        case GGML_OP_CONV_2D_DW:
            return ggml_sycl_supports_conv2d_dw(op);
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_sycl_supports_conv2d_transpose(op);
        case GGML_OP_CONV_TRANSPOSE_1D:
            return ggml_sycl_supports_conv_transpose_1d(op);
        case GGML_OP_COL2IM_1D:
            return ggml_sycl_supports_col2im_1d(op);
        case GGML_OP_POOL_1D:
            return ggml_sycl_supports_pool1d(op);
        case GGML_OP_POOL_2D:
            return ggml_sycl_supports_pool2d(op);
        case GGML_OP_UPSCALE:
            return ggml_sycl_supports_upscale(op);
        case GGML_OP_CONV_3D:
            return ggml_sycl_supports_conv3d(op);
        case GGML_OP_GET_ROWS:
            return ggml_sycl_supports_get_rows(op);
        case GGML_OP_SET_ROWS:
            return ggml_sycl_supports_set_rows(op);
        case GGML_OP_SET:
            return ggml_sycl_supports_set(op);
        case GGML_OP_ADD_ID:
            return ggml_sycl_supports_add_id(op);
        case GGML_OP_DIAG:
            return ggml_sycl_supports_diag(op);
        case GGML_OP_PAD:
            return ggml_sycl_supports_pad(op);
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_sycl_supports_pad_reflect_1d(op);
        case GGML_OP_ROLL:
            return ggml_sycl_supports_roll(op);
        case GGML_OP_TIMESTEP_EMBEDDING:
            return ggml_sycl_supports_timestep_embedding(op);
        case GGML_OP_TRI:
            return ggml_sycl_supports_tri(op);
        case GGML_OP_OPT_STEP_SGD:
            return ggml_sycl_supports_opt_step_sgd(op);
        case GGML_OP_OPT_STEP_ADAMW:
            return ggml_sycl_supports_opt_step_adamw(op);
        case GGML_OP_FILL:
            return ggml_sycl_supports_fill(op);
        case GGML_OP_SUM:
            return ggml_sycl_supports_sum(op);
        case GGML_OP_SUM_ROWS:
            return ggml_sycl_supports_sum_rows(op);
        case GGML_OP_MEAN:
            return ggml_sycl_supports_mean(op);
        case GGML_OP_ACC:
            return ggml_sycl_supports_acc(op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
            return ggml_sycl_supports_cross_entropy_loss(op);
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return ggml_sycl_supports_cross_entropy_loss_back(op);
        case GGML_OP_CUMSUM:
            return ggml_sycl_supports_cumsum(op);
        case GGML_OP_SOLVE_TRI:
            return ggml_sycl_supports_solve_tri(op);
        case GGML_OP_DSV4_HC_COMB:
            return ggml_sycl_supports_dsv4_hc_comb(op);
        case GGML_OP_DSV4_HC_PRE:
            return ggml_sycl_supports_dsv4_hc_pre(op);
        case GGML_OP_DSV4_HC_POST:
            return ggml_sycl_supports_dsv4_hc_post(op);
        case GGML_OP_SSM_CONV:
            return ggml_sycl_supports_ssm_conv(op);
        case GGML_OP_SSM_SCAN:
            return ggml_sycl_supports_ssm_scan(op);
        case GGML_OP_RWKV_WKV6:
            return ggml_sycl_supports_rwkv_wkv6(op);
        case GGML_OP_GATED_LINEAR_ATTN:
            return ggml_sycl_supports_gated_linear_attn(op);
        case GGML_OP_RWKV_WKV7:
            return ggml_sycl_supports_rwkv_wkv7(op);
        case GGML_OP_GATED_DELTA_NET:
            return ggml_sycl_supports_gated_delta_net(op);
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_sycl_supports_lightning_indexer(op);
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
