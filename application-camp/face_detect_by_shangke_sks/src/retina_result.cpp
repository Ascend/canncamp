#include <vector>
#include "commdef.h"
#include "process_output_tensor.h"
#include "anchor_generator.h"
#include <map>

using std::vector;
using std::map;

void result_to_faces( vector<vector<float>> &faces,
                      const map<string, shared_ptr<CProcessOutputTensor>>& result );

void output_2_face_rect( vector<vector<float>>& rects, const vector<vector<float>>& infer_outputs, int input_wh )
{
    do
    {
        BREAK_ON_FALSE( infer_outputs.size() == 9 );
        map<string, shared_ptr<CProcessOutputTensor>> infer_output_tensors;
        int i = 0;
        // 模型输出结果竟然不能获得每一层的维度信息、名称，好像只能通过顺序来判断是哪一个输出层。
        int wh = (int)(float(input_wh) / 32 + 0.99f);
        infer_output_tensors["face_rpn_bbox_pred_stride32"].reset( new CProcessOutputTensor( {1,8,wh,wh}, infer_outputs[i++] ) );
        infer_output_tensors["face_rpn_landmark_pred_stride32"].reset( new CProcessOutputTensor( {1,20,wh,wh}, infer_outputs[i++] ) );
        infer_output_tensors["face_rpn_cls_prob_reshape_stride32"].reset( new CProcessOutputTensor( {1,4,wh,wh}, infer_outputs[i++] ) );
        wh = (int)(float(input_wh) / 16 + 0.99f);
        infer_output_tensors["face_rpn_bbox_pred_stride16"].reset( new CProcessOutputTensor( {1,8,wh,wh}, infer_outputs[i++] ) );
        infer_output_tensors["face_rpn_landmark_pred_stride16"].reset( new CProcessOutputTensor( {1,20,wh,wh}, infer_outputs[i++] ) );
        infer_output_tensors["face_rpn_cls_prob_reshape_stride16"].reset( new CProcessOutputTensor( {1,4,wh,wh}, infer_outputs[i++] ) );
        wh = (int)(float(input_wh) / 8 + 0.99f);
        infer_output_tensors["face_rpn_bbox_pred_stride8"].reset( new CProcessOutputTensor( {1,8,wh,wh}, infer_outputs[i++] ) );
        infer_output_tensors["face_rpn_landmark_pred_stride8"].reset( new CProcessOutputTensor( {1,20,wh,wh}, infer_outputs[i++] ) );
        infer_output_tensors["face_rpn_cls_prob_reshape_stride8"].reset( new CProcessOutputTensor( {1,4,wh,wh}, infer_outputs[i++] ) );

        result_to_faces( rects, infer_output_tensors );

    } while ( false );

}

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx][0], boxes[tmp_i][0] );
            float inter_y1 = std::max( boxes[good_idx][1], boxes[tmp_i][1] );
            float inter_x2 = std::min( boxes[good_idx][2], boxes[tmp_i][2] );
            float inter_y2 = std::min( boxes[good_idx][3], boxes[tmp_i][3] );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}

void result_to_faces(vector<vector<float>> &faces, const map<string, shared_ptr<CProcessOutputTensor>> &result)
{
    vector<AnchorGenerator> anchor_generator;
    vector<int> feat_stride_fpn{32, 16, 8};
    map<int, AnchorCfg> m_anchor_cfg{{32, AnchorCfg(vector<float>{32, 16}, vector<float>{1}, 16)},
                                     {16, AnchorCfg(vector<float>{8, 4}, vector<float>{1}, 16)},
                                     {8, AnchorCfg(vector<float>{2, 1}, vector<float>{1}, 16)}};
    for (size_t i = 0; i < feat_stride_fpn.size(); i++)
    {
        AnchorGenerator tmp;
        tmp.Init(feat_stride_fpn[i], m_anchor_cfg[feat_stride_fpn[i]], false);
        anchor_generator.push_back(tmp);
    }

    vector<Anchor> proposals;

    anchor_generator[0].FilterAnchor(result.at("face_rpn_cls_prob_reshape_stride32").get(),
                                     result.at("face_rpn_bbox_pred_stride32").get(),
                                     result.at("face_rpn_landmark_pred_stride32").get(),
                                     proposals);
    anchor_generator[1].FilterAnchor(result.at("face_rpn_cls_prob_reshape_stride16").get(),
                                     result.at("face_rpn_bbox_pred_stride16").get(),
                                     result.at("face_rpn_landmark_pred_stride16").get(),
                                     proposals);
    anchor_generator[2].FilterAnchor(result.at("face_rpn_cls_prob_reshape_stride8").get(),
                                     result.at("face_rpn_bbox_pred_stride8").get(),
                                     result.at("face_rpn_landmark_pred_stride8").get(),
                                     proposals);

    vector<Anchor> results;
    nms_cpu(proposals, 0.4f, results);
    for (auto item : results)
    {
        float x = item[0];
        float y = item[1];
        float w = item[2] - item[0];
        float h = item[3] - item[1];

        faces.push_back({x, y, w, h});
    }
}
