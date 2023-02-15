#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
roi_extractor
"""
import functools
import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.roi_align import roi_align_compute


def ceil_div(value, factor):
    """
    ceil div
    """
    return (value + factor - 1) // factor


# 16K size
UB_30K_SIZE = 150 * 1024
BLOCK_BIT_SIZE = 32
BIT_EACH_BYTE = 8
ALIGN_LEN = 128
C0_SIZE_FP16 = 16
C0_SIZE_FP32 = 8
REPEAT_FP16 = ceil_div(ALIGN_LEN, 128)
REPEAT_FP32 = ceil_div(ALIGN_LEN, 64)


def _tf_n52n8(tik_instance, rois_ub, rois_n5, block_num):
    """
    transform ROIS form N5 to N8
    """
    with tik_instance.for_range(0, block_num) as rois_num:
        rois_ub[rois_num, 0].set_as(rois_n5[rois_num, 0])
        rois_ub[rois_num, 1].set_as(rois_n5[rois_num, 1])
        rois_ub[rois_num, 2].set_as(rois_n5[rois_num, 2])
        rois_ub[rois_num, 3].set_as(rois_n5[rois_num, 3])
        rois_ub[rois_num, 4].set_as(rois_n5[rois_num, 4])


# 'pylint: disable=too-many-statements,too-many-locals,too-many-branches
# 'pylint: disable=no-member,too-many-arguments
def _get_roi_align_perf_scale_for_zero(tik_instance, proposal, proposals_ub_x0,
                                       proposals_ub_y0, proposals_ub_x1,
                                       proposals_ub_y1, scale, pool_h, pool_w,
                                       sample_num, roi_end_mode, aligned, dtype):
    """
    get satart point, bin_size and sample number
    """
    proposal_num_128 = 128
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2

    roi_h_fp32 = tik_instance.Tensor(
        dtype, [128], name="roi_h_fp32", scope=tbe_platform.scope_ubuf)
    roi_w_fp32 = tik_instance.Tensor(
        dtype, [128], name="roi_w_fp32", scope=tbe_platform.scope_ubuf)

    roi_fp32_fm_index = tik_instance.Tensor(
        dtype, [128], name="roi_fp32_fm_index", scope=tbe_platform.scope_ubuf)
    roi_int32_fm_index = tik_instance.Tensor(
        "int32", [128], name="roi_int32_fm_index", scope=tbe_platform.scope_ubuf)

    suppot_vconv = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32c")

    if not suppot_vconv and dtype == "float32":
        roi_fp16_pos = tik_instance.Tensor(
            "float16", proposal.shape, name="roi_fp16_pos", scope=tbe_platform.scope_ubuf)
        roi_fp16_fm_index = tik_instance.Tensor(
            "float16", [128], name="roi_fp16_fm_index", scope=tbe_platform.scope_ubuf)
        tik_instance.vec_conv(64, "", roi_fp16_pos[0, 0], proposal[0, 0],
                              (128 * 8) // 64, 4, 8)

        with tik_instance.for_range(0, 128) as idx:
            roi_fp16_fm_index[idx].set_as(roi_fp16_pos[idx, 0])

        tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                              roi_fp16_fm_index[0], 2, 8, 4)
    else:
        with tik_instance.for_range(0, 128) as idx:
            roi_fp32_fm_index[idx].set_as(proposal[idx, 0])

        tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                              roi_fp32_fm_index[0], 2, 8, 8 // dtype_num)

    tik_instance.vec_muls(64 * dtype_num, proposals_ub_x0[0, 0],
                          proposals_ub_x0[0, 0],
                          scale, 128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, proposals_ub_y0[0, 0],
                          proposals_ub_y0[0, 0],
                          scale, 128 * 2 // 128 // dtype_num, 8, 8)

    if roi_end_mode == 1:
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_x1[0, 0],
                              proposals_ub_x1[0, 0], 1,
                              128 * 2 // 128 // dtype_num, 8, 8)
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_y1[0, 0],
                              proposals_ub_y1[0, 0], 1,
                              128 * 2 // 128 // dtype_num, 8, 8)

    tik_instance.vec_muls(64 * dtype_num, proposals_ub_x1[0, 0],
                          proposals_ub_x1[0, 0],
                          scale, 128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, proposals_ub_y1[0, 0],
                          proposals_ub_y1[0, 0],
                          scale, 128 * 2 // 128 // dtype_num, 8, 8)

    # for pytorch/mmcv
    if aligned:
        offset = -0.5
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_x0[0, 0],
                              proposals_ub_x0[0, 0], offset,
                              128 * 2 // 128 // dtype_num, 8, 8)
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_y0[0, 0],
                              proposals_ub_y0[0, 0], offset,
                              128 * 2 // 128 // dtype_num, 8, 8)
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_x1[0, 0],
                              proposals_ub_x1[0, 0], offset,
                              128 * 2 // 128 // dtype_num, 8, 8)
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_y1[0, 0],
                              proposals_ub_y1[0, 0], offset,
                              128 * 2 // 128 // dtype_num, 8, 8)

    roi_h_1to8 = tik_instance.Tensor(
        dtype, [128, 1], name="roi_h_1to8", scope=tbe_platform.scope_ubuf)
    roi_w_1to8 = tik_instance.Tensor(
        dtype, [128, 1], name="roi_w_1to8", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_sub(64 * dtype_num, roi_h_1to8, proposals_ub_y1[0, 0],
                         proposals_ub_y0[0, 0], 128 * 2 // 128 // dtype_num,
                         8, 8, 8)
    tik_instance.vec_sub(64 * dtype_num, roi_w_1to8, proposals_ub_x1[0, 0],
                         proposals_ub_x0[0, 0], 128 * 2 // 128 // dtype_num,
                         8, 8, 8)

    const_mode = tik_instance.Tensor(
        dtype, [128, 1], name="const_mode", scope=tbe_platform.scope_ubuf)
    tik_instance.vec_dup(64 * dtype_num, const_mode, 1 - roi_end_mode,
                         2 // dtype_num, 8)

    # compare roi_width adn roi_height to 1-mode (1 or 0)
    tik_instance.vec_max(64 * dtype_num, roi_w_1to8, roi_w_1to8, const_mode,
                         128 * 2 // 128 // dtype_num, 8, 8, 0)
    tik_instance.vec_max(64 * dtype_num, roi_h_1to8, roi_h_1to8, const_mode,
                         128 * 2 // 128 // dtype_num, 8, 8, 0)

    with tik_instance.for_range(0, roi_w_fp32.shape[0]) as i:
        roi_w_fp32[i].set_as(roi_w_1to8[i, 0])
        roi_h_fp32[i].set_as(roi_h_1to8[i, 0])

    # Declare roi_bin_size tik_instance.Tensor
    roi_bin_h_fp32_value = tik_instance.Tensor(
        dtype, [128], name="roi_bin_h_fp32_value", scope=tbe_platform.scope_ubuf)
    roi_bin_w_fp32_value = tik_instance.Tensor(
        dtype, [128], name="roi_bin_w_fp32_value", scope=tbe_platform.scope_ubuf)

    grid_w_fp32 = tik_instance.Tensor(
        dtype, [proposal_num_128], name="grid_w_fp32", scope=tbe_platform.scope_ubuf)
    grid_h_fp32 = tik_instance.Tensor(
        dtype, [proposal_num_128], name="grid_h_fp32", scope=tbe_platform.scope_ubuf)

    grid_w_fp16 = tik_instance.Tensor(
        "float16", [proposal_num_128], name="grid_w_fp16", scope=tbe_platform.scope_ubuf)
    grid_h_fp16 = tik_instance.Tensor(
        "float16", [proposal_num_128], name="grid_h_fp16", scope=tbe_platform.scope_ubuf)

    grid_w_int32 = tik_instance.Tensor(
        "int32", [proposal_num_128], name="grid_w_int32", scope=tbe_platform.scope_ubuf)
    grid_h_int32 = tik_instance.Tensor(
        "int32", [proposal_num_128], name="grid_h_int32", scope=tbe_platform.scope_ubuf)

    # bin size
    tik_instance.vec_muls(64 * dtype_num, roi_bin_h_fp32_value[:],
                          roi_h_fp32[:], 1.0 / pool_h,
                          proposal_num_128 * 2 // dtype_num // 128, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, roi_bin_w_fp32_value[:],
                          roi_w_fp32[:], 1.0 / pool_w,
                          proposal_num_128 * 2 // dtype_num // 128, 8, 8)
    suppot_vconv = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32c")
    if sample_num <= 0:
        if suppot_vconv is False and dtype == "float32":
            roi_bin_w_fp16_value = tik_instance.Tensor(
                "float16", [128],
                name="roi_bin_w_fp16_value",
                scope=tbe_platform.scope_ubuf)
            roi_bin_h_fp16_value = tik_instance.Tensor(
                "float16", [128],
                name="roi_bin_h_fp16_value",
                scope=tbe_platform.scope_ubuf)
            tik_instance.vec_conv(64 * dtype_num, "", roi_bin_w_fp16_value,
                                  roi_bin_w_fp32_value, 2, 4, 8)
            tik_instance.vec_conv(64 * dtype_num, "", roi_bin_h_fp16_value,
                                  roi_bin_h_fp32_value, 2, 4, 8)

            tik_instance.vec_conv(64, "ceiling", grid_w_int32,
                                  roi_bin_w_fp16_value, 2, 8, 4)
            tik_instance.vec_conv(64, "ceiling", grid_h_int32,
                                  roi_bin_h_fp16_value, 2, 8, 4)
        else:
            tik_instance.vec_conv(64, "ceiling", grid_w_int32,
                                  roi_bin_w_fp32_value, 2, 8,
                                  8 // dtype_num)
            tik_instance.vec_conv(64, "ceiling", grid_h_int32,
                                  roi_bin_h_fp32_value, 2, 8,
                                  8 // dtype_num)

        if suppot_vconv is False and dtype == "float32":
            tik_instance.vec_conv(64 * dtype_num, "", grid_w_fp16, grid_w_int32,
                                  2 // dtype_num, 4, 8, 1.0)
            tik_instance.vec_conv(64 * dtype_num, "", grid_h_fp16, grid_h_int32,
                                  2 // dtype_num, 4, 8, 1.0)
            tik_instance.vec_conv(64 * dtype_num, "", grid_w_fp32, grid_w_fp16,
                                  2 // dtype_num, 8, 4)
            tik_instance.vec_conv(64 * dtype_num, "", grid_h_fp32, grid_h_fp16,
                                  2 // dtype_num, 8, 4)
        else:
            if dtype == "float32":
                tik_instance.vec_conv(64, "", grid_w_fp32, grid_w_int32, 2,
                                      8 // dtype_num, 8)
                tik_instance.vec_conv(64, "", grid_h_fp32, grid_h_int32, 2,
                                      8 // dtype_num, 8)
            else:
                tik_instance.vec_conv(64, "", grid_w_fp32, grid_w_int32, 2,
                                      8 // dtype_num, 8, 1.0)
                tik_instance.vec_conv(64, "", grid_h_fp32, grid_h_int32, 2,
                                      8 // dtype_num, 8, 1.0)

    else:
        tik_instance.vec_dup(64, grid_w_int32, sample_num, 2, 8)
        tik_instance.vec_dup(64, grid_h_int32, sample_num, 2, 8)
        tik_instance.vec_dup(64 * dtype_num, grid_w_fp32,
                             sample_num, 2 // dtype_num, 8)
        tik_instance.vec_dup(64 * dtype_num, grid_h_fp32,
                             sample_num, 2 // dtype_num, 8)

    return tik_instance, roi_bin_h_fp32_value, \
           roi_bin_w_fp32_value, \
           proposals_ub_x0, proposals_ub_y0, \
           grid_w_int32, grid_h_int32, grid_w_fp32, \
           grid_h_fp32, roi_int32_fm_index


def extract_roi(tik_instance, roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, roi_num):
    """
    extract roi
    """
    with tik_instance.for_range(0, roi_num) as idx:
        x0_ub[idx].set_as(roi_ub[idx, 1])
        y0_ub[idx].set_as(roi_ub[idx, 2])
        x1_ub[idx].set_as(roi_ub[idx, 3])
        y1_ub[idx].set_as(roi_ub[idx, 4])


# 'pylint: disable=too-many-arguments
def roi_align_tik(tik_instance, feature_map, rois, roisn, ret,
                  scale, pool_h, pool_w, sample_ratio, roi_end_mode=0, aligned=True):
    """
    roi align func
    """
    rois_shape = rois.shape
    dtype = feature_map.dtype
    feature_shape = feature_map.shape
    roisn_dtype = roisn.dtype

    fm_c1 = feature_shape[1]
    proposal_num = tik_instance.Scalar("int32", name="proposal_num")
    proposal_num.set_as(roisn[0][0])
    grid_curr_h = tik_instance.Scalar(dtype="int32")
    grid_curr_w = tik_instance.Scalar(dtype="int32")
    block_num = proposal_num
    if dtype == "float32":
        n_bust = 2
    else:
        n_bust = 1
    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    roisn_ub = tik_instance.Tensor(roisn_dtype, [1, 16], \
                                   name="roisn_ub", scope=tbe_platform.scope_ubuf)
    tik_instance.data_move(roisn_ub, roisn, 0, 1, 1, 0, 0)
    # every block, process 128 rois
    with tik_instance.for_range(0, 1): # multi core close
        block_i = tik_instance.Scalar(dtype="int32")
        block_i.set_as(0)

        rois_ub = tik_instance.Tensor(
            dtype, [128, 8], name="rois_ub", scope=tbe_platform.scope_ubuf)
        proposals_ub_x0 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_x0", scope=tbe_platform.scope_ubuf)
        proposals_ub_y0 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_y0", scope=tbe_platform.scope_ubuf)
        proposals_ub_x1 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_x1", scope=tbe_platform.scope_ubuf)
        proposals_ub_y1 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_y1", scope=tbe_platform.scope_ubuf)

        if dtype == "float32":
            tik_instance.vector_dup(64, rois_ub, 0.0, 16, 1, 8)
        else:
            tik_instance.vector_dup(128, rois_ub, 0.0, 8, 1, 8)

        rois_valid = tik_instance.Scalar(dtype="int32", init_value=block_num)
        with tik_instance.if_scope(
                block_i == ((proposal_num + (block_num - 1)) // block_num - 1)):
            rois_valid.set_as(proposal_num - block_i * block_num)

        with tik_instance.if_scope(rois_valid != 0):
            with tik_instance.for_range(
                    0, (rois_valid + (128 - 1))//128) as roi_128_number:
                rois_valid_in_block = \
                    tik_instance.Scalar(dtype="int32", init_value=128)
                with tik_instance.if_scope(
                        roi_128_number == ((rois_valid + (128 - 1))//128 - 1)):
                    rois_valid_in_block.set_as(
                        rois_valid - roi_128_number * 128)

                if rois_shape[1] == 5:
                    rois_ub_n5 = tik_instance.Tensor(
                        dtype, [128, 5], name="rois_ub_n5",
                        scope=tbe_platform.scope_ubuf)
                    tik_instance.data_move(rois_ub_n5[0, 0],
                                           rois[block_i * block_num +
                                                roi_128_number * 128, 0],
                                           0, 1,
                                           40 * n_bust, 0, 0)
                    _tf_n52n8(tik_instance, rois_ub, rois_ub_n5, 128)
                else:
                    tik_instance.data_move(rois_ub[0, 0],
                                           rois[block_i * block_num +
                                                roi_128_number * 128, 0],
                                           0, 1,
                                           64 * n_bust, 0, 0)

                extract_roi(tik_instance, rois_ub, proposals_ub_x0, proposals_ub_y0, proposals_ub_x1, proposals_ub_y1,
                            128)

                tik_instance, roi_bin_h_fp32_value, \
                roi_bin_w_fp32_value, \
                proposals_ub_x0, proposals_ub_y0, \
                grid_w_int32, grid_h_int32, grid_w_fp32, \
                grid_h_fp32, roi_int32_fm_index = \
                    _get_roi_align_perf_scale_for_zero(tik_instance, rois_ub,
                                                       proposals_ub_x0,
                                                       proposals_ub_y0,
                                                       proposals_ub_x1,
                                                       proposals_ub_y1,
                                                       scale, pool_h, pool_w,
                                                       sample_ratio,
                                                       roi_end_mode,
                                                       aligned,
                                                       dtype)

                w_number = 0
                w_number_ub = 0
                ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - UB_30K_SIZE
                feature_map_to_ub_verify = ub_size_bytes // \
                                           (fm_c1 * feature_shape[2] * feature_shape[3] * 16 * n_bust * 2)
                feature_map_to_l1_verify = \
                    l1_size // (fm_c1 * feature_shape[2] * \
                                feature_shape[3] * 16 * n_bust * 2)
                if feature_map_to_ub_verify == 0 and feature_map_to_l1_verify == 0:
                    w_number_ub = ub_size_bytes // \
                                  (feature_shape[1] * feature_shape[3] *
                                   feature_shape[4] * n_bust * 2)
                if feature_map_to_ub_verify == 0 and \
                        feature_map_to_l1_verify == 0 and w_number_ub == 0:
                    if (feature_shape[3] - 1) * n_bust < 65535:
                        w_number = l1_size // (feature_shape[1] * \
                                               feature_shape[3] * \
                                               feature_shape[4] * n_bust * 2)

                roi_align_compute(
                    tik_instance, feature_map, ret, proposals_ub_x0,
                    proposals_ub_y0, pool_h, pool_w, dtype, roi_128_number,
                    rois_valid_in_block,
                    feature_shape, grid_curr_h, grid_curr_w, fm_c1, n_bust,
                    block_i, block_num, roi_int32_fm_index, grid_h_int32,
                    grid_w_int32, grid_h_fp32, grid_w_fp32,
                    roi_bin_h_fp32_value,
                    roi_bin_w_fp32_value, w_number,
                    feature_map_to_l1_verify, feature_map_to_ub_verify,
                    w_number_ub)

    return tik_instance


# 'pylint: disable=unused-argument
class RoiExtractor:
    """
    roi_extractor op
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, feats, rois, index, roi_feats,
                 finest_scale=56, roi_scale_factor=0,
                 spatial_scale=None, pooled_h=7, pooled_w=7,
                 sample_num=0, pool_mode='avg',
                 aligned=False, kernel_name="roi_extractor"):
        if pool_mode != 'avg':
            raise RuntimeError("pool_mode only support avg")

        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name

        self.finest_scale = finest_scale

        # RoiAlign Parameters
        self.roi_scale_factor = roi_scale_factor
        self.output_size = pooled_h
        self.spatial_scale = spatial_scale
        self.sample_ratio = sample_num
        self.aligned = aligned

        # golobal buffer
        self.rois_shape = rois.get("shape")
        self.rois_dtype = rois.get("dtype")
        self.x_dtype = feats[0].get("dtype")
        self.y_shape = roi_feats.get("shape")
        self.rois_total_num = self.rois_shape[0]
        self.num_levels = len(feats)
        self.pow_table = [2 ** i * self.finest_scale for i in range(self.num_levels + 1)]
        self.index_arr = [i for i in range(self.num_levels)]

        if self.rois_shape[1] not in (4, 5):
            raise RuntimeError("rois last dim should be 4 or 5")

        self.data_x = []
        for i, tensor_dict in enumerate(feats):
            x_shape = tensor_dict.get("shape")
            self.data_x.append(self.tik_instance.Tensor(self.x_dtype, x_shape, name="x_%d" % i, scope=tik.scope_gm))
        self.data_rois = self.tik_instance.Tensor(self.rois_dtype, self.rois_shape, name="rois", scope=tik.scope_gm)
        self.data_y = self.tik_instance.Tensor(self.x_dtype, (functools.reduce(lambda x, y: x * y, self.y_shape), ),
                                               name="roi_feats", scope=tik.scope_gm)

        # multi core
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    def map_roi_levels(self, target_lvls, proc_num, roi_offset=0):
        """
        `calculate scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))`
        `:return: target_lvls`
        """
        dtype_bytes_size = tbe_platform.get_bit_len(self.rois_dtype) // BIT_EACH_BYTE

        x1_fp16 = self.tik_instance.Tensor("float16", (ALIGN_LEN,), name="rois_x1_fp16", scope=tik.scope_ubuf)
        y1_fp16 = self.tik_instance.Tensor("float16", (ALIGN_LEN,), name="rois_y1_fp16", scope=tik.scope_ubuf)
        x2_fp16 = self.tik_instance.Tensor("float16", (ALIGN_LEN,), name="rois_x2_fp16", scope=tik.scope_ubuf)
        y2_fp16 = self.tik_instance.Tensor("float16", (ALIGN_LEN,), name="rois_y2_fp16", scope=tik.scope_ubuf)
        x1_fp32 = self.tik_instance.Tensor("float32", (ALIGN_LEN,), name="rois_x1_fp32_ub", scope=tik.scope_ubuf)
        y1_fp32 = self.tik_instance.Tensor("float32", (ALIGN_LEN,), name="rois_y1_fp32_ub", scope=tik.scope_ubuf)
        x2_fp32 = self.tik_instance.Tensor("float32", (ALIGN_LEN,), name="rois_x2_fp32_ub", scope=tik.scope_ubuf)
        y2_fp32 = self.tik_instance.Tensor("float32", (ALIGN_LEN,), name="rois_y2_fp32_ub", scope=tik.scope_ubuf)
        lvl_max = self.tik_instance.Tensor("float16", (128,), name="lvl_max_ub", scope=tik.scope_ubuf)
        one_ub = self.tik_instance.Tensor("float16", (128,), name="one_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik_instance.Tensor("float16", (128,), name="zero_ub", scope=tik.scope_ubuf)
        work_tensor0 = self.tik_instance.Tensor("float32", (4 * ALIGN_LEN,),
                                                name="work_tensor0_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, lvl_max, self.num_levels - 1, 1, 1, 8)
        self.tik_instance.vector_dup(128, one_ub, 1, 1, 1, 8)
        self.tik_instance.vector_dup(128, zero_ub, 0, 1, 1, 8)

        loop = proc_num // ALIGN_LEN
        tail = proc_num % ALIGN_LEN

        rois_ub = self.tik_instance.Tensor(self.rois_dtype, (ALIGN_LEN, 8),
                                           name="rois_ub", scope=tik.scope_ubuf)
        rois_ub_n5 = self.tik_instance.Tensor(self.rois_dtype, (ALIGN_LEN, 5),
                                              name="rois_ub_n5", scope=tik.scope_ubuf)
        if self.rois_dtype == "float32":
            n_burst = 2
        else:
            n_burst = 1

        def inner_compute(offset):
            """
            compute roi area
            """
            if self.rois_dtype == "float16":
                extract_roi(self.tik_instance, rois_ub, x1_fp16, y1_fp16, x2_fp16, y2_fp16, 128)
                self.tik_instance.vec_conv(64, "none", x1_fp32, x1_fp16, REPEAT_FP32, 8, 4)  # fp16->fp32
                self.tik_instance.vec_conv(64, "none", y1_fp32, y1_fp16, REPEAT_FP32, 8, 4)
                self.tik_instance.vec_conv(64, "none", x2_fp32, x2_fp16, REPEAT_FP32, 8, 4)
                self.tik_instance.vec_conv(64, "none", y2_fp32, y2_fp16, REPEAT_FP32, 8, 4)
            else:
                extract_roi(self.tik_instance, rois_ub, x1_fp32, y1_fp32, x2_fp32, y2_fp32, 128)

            # calc levels
            self.tik_instance.vec_sub(64, x1_fp32, x2_fp32, x1_fp32, REPEAT_FP32, 8, 8, 8)  # x2_fp32 - x1_fp32
            self.tik_instance.vec_sub(64, y1_fp32, y2_fp32, y1_fp32, REPEAT_FP32, 8, 8, 8)  # y2_fp32 - y1_fp32
            self.tik_instance.vec_mul(64, y1_fp32, x1_fp32, y1_fp32, REPEAT_FP32, 8, 8, 8)
            self.tik_instance.vec_rsqrt_high_preci(64, x1_fp32, y1_fp32, work_tensor0[0:], REPEAT_FP32, 8, 8)
            self.tik_instance.vec_rec_high_preci(64, y1_fp32, x1_fp32, work_tensor0[0:], REPEAT_FP32, 8, 8)  # sqrt(x)
            self.tik_instance.vec_conv(64, "none", target_lvls[offset:offset + ALIGN_LEN], y1_fp32, REPEAT_FP32, 4, 8)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as loop_i:
                self.tik_instance.data_move(rois_ub_n5[0, 0], self.data_rois[loop_i * ALIGN_LEN + roi_offset, 0],
                                            0, 1, 40 * n_burst, 0, 0)
                _tf_n52n8(self.tik_instance, rois_ub, rois_ub_n5, ALIGN_LEN)
                inner_compute(loop_i * ALIGN_LEN)
        if tail > 0:
            self.tik_instance.data_move(rois_ub_n5[0, 0], self.data_rois[loop * ALIGN_LEN + roi_offset, 0],
                                        0, 1, 40 * n_burst, 0, 0)
            _tf_n52n8(self.tik_instance, rois_ub, rois_ub_n5, tail)
            inner_compute(loop * ALIGN_LEN)

    def tik_vand(self, dst, src0, src1, data_len):
        """
        vand
        """
        offset = 0
        loop = data_len // (128 * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as loop_i:
                offset = loop_i * 128 * 255
                self.tik_instance.vec_and(128, dst[offset], src0[offset], src1[offset], 255, 8, 8, 8)
        repeat_t = data_len // 128
        if repeat_t > 0:
            self.tik_instance.vec_and(128, dst[offset], src0[offset], src1[offset], repeat_t, 8, 8, 8)
        offset += repeat_t * 128
        tail = data_len % 128
        if tail > 0:
            self.tik_instance.vec_and(tail, dst[offset], src0[offset], src1[offset], 1, 8, 8, 8)

    def where_and_nonzero(self, target_lvls, index_reg, inds, lvl, proc_num):
        """
        select roi
        """
        align_len = target_lvls.shape[0]
        inds_buf = self.tik_instance.Tensor("int32", (align_len, ),
                                            name="inds_buf", scope=tik.scope_ubuf)
        i_ub = self.tik_instance.Tensor("float16", (128,), name="i_ub", scope=tik.scope_ubuf)
        i1_ub = self.tik_instance.Tensor("float16", (128,), name="i1_ub", scope=tik.scope_ubuf)
        cmp_bit_size = 16
        cmp_ub = self.tik_instance.Tensor("uint16", (align_len // cmp_bit_size, ),
                                          name="cmp_ub", scope=tik.scope_ubuf)
        cmp1_ub = self.tik_instance.Tensor("uint16", (align_len // cmp_bit_size, ),
                                           name="cmp1_ub", scope=tik.scope_ubuf)
        one_ub = self.tik_instance.Tensor("float16", (128,), name="one_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik_instance.Tensor("float16", (128,), name="zero_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, one_ub, 1, 1, 1, 8)
        self.tik_instance.vector_dup(128, zero_ub, 0, 1, 1, 8)
        for i in self.index_arr:
            with self.tik_instance.if_scope(lvl == i):
                self.tik_instance.vector_dup(128, i_ub, self.pow_table[i], 1, 1, 8)
                self.tik_instance.vector_dup(128, i1_ub, self.pow_table[i + 1], 1, 1, 8)
            with self.tik_instance.else_scope():
                pass

        repeat = align_len // 128
        with self.tik_instance.if_scope(lvl == 0):
            self.tik_instance.vec_cmpv_le(cmp_ub, target_lvls, i1_ub, repeat, 8, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(lvl == self.num_levels - 1):
                self.tik_instance.vec_cmpv_gt(cmp_ub, target_lvls, i_ub, repeat, 8, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_cmpv_gt(cmp_ub, target_lvls, i_ub, repeat, 8, 0)
                self.tik_instance.vec_cmpv_le(cmp1_ub, target_lvls, i1_ub, repeat, 8, 0)
                self.tik_vand(cmp_ub, cmp_ub, cmp1_ub, align_len // cmp_bit_size)

        with self.tik_instance.for_range(0, repeat) as i:
            self.tik_instance.vec_sel(128, 0, i_ub, cmp_ub[128 // cmp_bit_size * i], one_ub, zero_ub, 1, 1, 8, 8)
            self.tik_instance.vec_conv(64, "floor", inds_buf[128 * i], i_ub, 2, 8, 4)  # fp16->i32
        idx = self.tik_instance.Scalar("int32", name="roi_idx")
        index_reg.set_as(0)
        with self.tik_instance.for_range(0, proc_num) as roi_i:
            idx.set_as(inds_buf[roi_i])
            with self.tik_instance.if_scope(idx != 0):
                inds[index_reg].set_as(roi_i)
                index_reg.set_as(index_reg + 1)

    def compute(self):
        """
        roi extractor compute
        """
        dtype_bytes_size = tbe_platform.get_bit_len(self.rois_dtype) // BIT_EACH_BYTE
        data_each_block = BLOCK_BIT_SIZE // dtype_bytes_size
        roi_elem_len = functools.reduce(lambda x, y: x * y, self.y_shape[1:])

        roi_buf = self.tik_instance.Tensor(self.rois_dtype, (self.num_levels * self.rois_total_num, 5), name="roi_buf",
                                           scope=tik.scope_gm, is_workspace=True)
        roi_feats_shape = [self.num_levels * self.y_shape[0]] + list(self.y_shape[1:])
        roi_feats_buf = self.tik_instance.Tensor(self.x_dtype, roi_feats_shape, name="roi_feats_buf",
                                                 scope=tik.scope_gm, is_workspace=True)

        with self.tik_instance.for_range(0, self.num_levels, block_num=self.num_levels) as block_idx:
            roisn_ub = self.tik_instance.Tensor("int32", (data_each_block, ), name="roisn_ub", scope=tik.scope_ubuf)
            inds = self.tik_instance.Tensor("int32", (ceil_div(self.rois_total_num, 128) * 128, ),
                                            name="inds_ub", scope=tik.scope_ubuf)
            index_reg = self.tik_instance.Scalar("int32", name="index_reg")  # valid rois number in current level
            idx = self.tik_instance.Scalar("int32", name="roi_idx")

            with self.tik_instance.new_stmt_scope():
                target_lvls = self.tik_instance.Tensor("float16", (ceil_div(self.rois_total_num, 128) * 128,),
                                                       name="target_lvls_ub", scope=tik.scope_ubuf)
                self.map_roi_levels(target_lvls, self.rois_total_num)
                self.where_and_nonzero(target_lvls, index_reg, inds, block_idx, self.rois_total_num)
                roisn_ub[0].set_as(index_reg)

                rois_ub = self.tik_instance.Tensor(self.rois_dtype, (data_each_block, ),
                                                   name="rois_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, index_reg) as roi_i:
                    idx.set_as(inds[roi_i])
                    self.tik_instance.data_move(rois_ub, self.data_rois[idx, 0], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(roi_buf[block_idx * self.y_shape[0] + roi_i, 0], rois_ub, 0, 1, 1, 0, 0)

            # call roi_align module
            with self.tik_instance.if_scope(index_reg > 0):
                for i in self.index_arr:
                    with self.tik_instance.if_scope(block_idx == i):
                        self.tik_instance = roi_align_tik(
                            self.tik_instance,
                            self.data_x[i],
                            roi_buf[block_idx * self.rois_total_num:(block_idx + 1) * self.rois_total_num, :],
                            roisn_ub,
                            roi_feats_buf[block_idx * self.y_shape[0]:(block_idx + 1) * self.y_shape[0], :, :, :, :],
                            self.spatial_scale[i],
                            self.output_size,
                            self.output_size,
                            self.sample_ratio,
                            0,
                            self.aligned
                        )
                    with self.tik_instance.else_scope():
                        pass

            with self.tik_instance.new_stmt_scope():
                out_ub = self.tik_instance.Tensor(self.rois_dtype, (roi_elem_len, ), name="out_ub",
                                                  scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, index_reg) as roi_i:
                    idx.set_as(inds[roi_i])
                    self.tik_instance.data_move(out_ub, roi_feats_buf[block_idx * self.y_shape[0] + roi_i, 0, 0, 0, 0],
                                                0, 1, ceil_div(roi_elem_len, data_each_block), 0, 0)
                    self.tik_instance.data_move(self.data_y[idx * roi_elem_len], out_ub, 0, 1,
                                                ceil_div(roi_elem_len, data_each_block), 0, 0)

        # build
        sch_list = [dx for dx in self.data_x] + [self.data_rois]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=sch_list, outputs=[self.data_y])
        return self.tik_instance

    def balance_compute(self, index):
        """
        fpn_roi extractor compute when soc_version is Ascend310P
        """
        index_ub, data_index = None, None
        if index is not None:
            index_shape = index.get("shape")
            index_dtype = index.get("dtype")
            data_index = self.tik_instance.Tensor(index_dtype, index_shape, name="index", scope=tik.scope_gm)

        dtype_bytes_size = tbe_platform.get_bit_len(self.rois_dtype) // BIT_EACH_BYTE
        data_each_block = BLOCK_BIT_SIZE // dtype_bytes_size
        roi_elem_len = functools.reduce(lambda x, y: x * y, self.y_shape[1:])

        roi_buf = self.tik_instance.Tensor(self.rois_dtype, (self.num_levels * self.rois_total_num, 5), name="roi_buf",
                                           scope=tik.scope_gm, is_workspace=True)
        roi_feats_shape = [self.num_levels * self.y_shape[0]] + list(self.y_shape[1:])
        roi_feats_buf = self.tik_instance.Tensor(self.x_dtype, roi_feats_shape, name="roi_feats_buf",
                                                 scope=tik.scope_gm, is_workspace=True)

        per_core_roi = math.ceil(self.rois_total_num / self.core_num)
        block_num = math.ceil(self.rois_total_num / per_core_roi)
        last_roi = self.rois_total_num - per_core_roi * (block_num - 1)

        with self.tik_instance.for_range(0, block_num, block_num=block_num) as block_idx:
            roisn_ub = self.tik_instance.Tensor("int32", (data_each_block,), name="roisn_ub", scope=tik.scope_ubuf)
            offset = block_idx * per_core_roi

            with self.tik_instance.if_scope(block_idx != block_num - 1):
                inds = self.tik_instance.Tensor("int32", (ceil_div(per_core_roi, 128) * 128,),
                                                name="inds_ub", scope=tik.scope_ubuf)
                if index is not None:
                    index_ub = self.tik_instance.Tensor("int32", (per_core_roi,), name="index_ub", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(index_ub, data_index[offset], 0, 1, (per_core_roi + 7) // 8, 0, 0)

                index_reg = self.tik_instance.Scalar("int32", name="index_reg")  # valid rois number in current level
                idx_tmp = self.tik_instance.Scalar("int32", name="roi_idx_tmp")
                idx = self.tik_instance.Scalar("int32", name="roi_idx")

                self.compute_per_core(per_core_roi, offset, index_reg, inds, roisn_ub, data_each_block, idx_tmp, idx,
                                      roi_buf, roi_feats_buf, roi_elem_len, index_ub)

            with self.tik_instance.else_scope():
                inds = self.tik_instance.Tensor("int32", (ceil_div(last_roi, 128) * 128,),
                                                name="inds_ub", scope=tik.scope_ubuf)
                if index is not None:
                    index_ub = self.tik_instance.Tensor("int32", (last_roi,), name="index_ub", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(index_ub, data_index[offset], 0, 1, (last_roi + 7) // 8, 0, 0)

                index_reg = self.tik_instance.Scalar("int32", name="index_reg")  # valid rois number in current level
                idx_tmp = self.tik_instance.Scalar("int32", name="roi_idx_tmp")
                idx = self.tik_instance.Scalar("int32", name="roi_idx")

                self.compute_per_core(last_roi, offset, index_reg, inds, roisn_ub, data_each_block, idx_tmp, idx,
                                      roi_buf, roi_feats_buf, roi_elem_len, index_ub)

        # build
        if index is not None:
            sch_list = [dx for dx in self.data_x] + [self.data_rois, data_index]
        else:
            sch_list = [dx for dx in self.data_x] + [self.data_rois]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=sch_list, outputs=[self.data_y])
        return self.tik_instance

    def compute_per_core(self, per_core_roi, offset, index_reg, inds, roisn_ub, data_each_block, idx_tmp, idx, roi_buf,
                         roi_feats_buf, roi_elem_len, index_ub):
        """
        roialign compute process
        """
        target_lvls = self.tik_instance.Tensor("float16", (ceil_div(per_core_roi, 128) * 128,),
                                               name="target_lvls_ub", scope=tik.scope_ubuf)
        self.map_roi_levels(target_lvls, per_core_roi, offset)

        out_ub = self.tik_instance.Tensor(self.rois_dtype, (roi_elem_len,), name="out_ub",
                                          scope=tik.scope_ubuf)
        for fm_idx in range(self.num_levels):
            self.where_and_nonzero(target_lvls, index_reg, inds, fm_idx, per_core_roi)
            roisn_ub[0].set_as(index_reg)

            rois_ub = self.tik_instance.Tensor(self.rois_dtype, (data_each_block,),
                                               name="rois_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, index_reg) as roi_i:
                idx_tmp.set_as(inds[roi_i])
                idx.set_as(idx_tmp + offset)
                self.tik_instance.data_move(rois_ub, self.data_rois[idx, 0], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(roi_buf[fm_idx * self.y_shape[0] + roi_i + offset, 0], rois_ub, 0,
                                            1, 1, 0, 0)

            # call roi_align module
            with self.tik_instance.if_scope(index_reg > 0):
                roi_align_tik(self.tik_instance, self.data_x[fm_idx],
                              roi_buf[fm_idx * self.rois_total_num + offset:fm_idx * self.rois_total_num + offset +
                                      per_core_roi, :], roisn_ub,
                              roi_feats_buf[fm_idx * self.y_shape[0] + offset:fm_idx * self.y_shape[0] + offset +
                                            per_core_roi, :, :, :, :], self.spatial_scale[fm_idx], self.output_size,
                              self.output_size, self.sample_ratio, 0, self.aligned)

            with self.tik_instance.for_range(0, index_reg) as roi_i:
                idx_tmp.set_as(inds[roi_i])
                if index_ub is not None:
                    idx.set_as(index_ub[idx_tmp])
                else:
                    idx.set_as(idx_tmp + offset)
                self.tik_instance.data_move(out_ub, roi_feats_buf[fm_idx * self.y_shape[0] + roi_i + offset,
                                                                  0, 0, 0, 0],
                                            0, 1, ceil_div(roi_elem_len, data_each_block), 0, 0)
                self.tik_instance.data_move(self.data_y[idx * roi_elem_len], out_ub, 0, 1,
                                            ceil_div(roi_elem_len, data_each_block), 0, 0)


# 'pylint: disable=unused-argument,too-many-arguments
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def roi_extractor(feats, rois, index=None, roi_feats=None,
                  finest_scale=56, roi_scale_factor=0, spatial_scale=None,
                  pooled_h=7, pooled_w=7, sample_num=0, pool_mode='avg', aligned=True, kernel_name="roi_extractor"):
    """
    roi extractor op
    """
    rpn_instance = RoiExtractor(feats, rois, index, roi_feats,
                                finest_scale, roi_scale_factor, spatial_scale,
                                pooled_h, pooled_w, sample_num, pool_mode, aligned, kernel_name)

    cce_product = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)

    if cce_product == tbe_platform.ASCEND_310P:
        return rpn_instance.balance_compute(index)

    return rpn_instance.compute()
