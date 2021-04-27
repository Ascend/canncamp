#include "anchor_generator.h"
#include <iostream>
#include "commdef.h"
#include "process_output_tensor.h"

AnchorGenerator::AnchorGenerator() {
  cls_threshold = 0.8f;
}

AnchorGenerator::~AnchorGenerator() {
}

// anchor plane
int AnchorGenerator::Generate(int fwidth, int fheight, int stride, float step, std::vector<int>& size, std::vector<float>& ratio, bool dense_anchor) {
  return 0;
}

// init different anchors
int AnchorGenerator::Init(int stride, const AnchorCfg& cfg, bool dense_anchor) {
  CRect2f base_anchor((float)0, (float)0, (float)cfg.BASE_SIZE-1, (float)cfg.BASE_SIZE-1);
  std::vector<CRect2f> ratio_anchors;
  // get ratio anchors
  _ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
  _scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);
  #if 0
  std::cout <<"ratio_anchors.size():" << ratio_anchors.size() <<std::endl;
  for(auto cf:ratio_anchors)
  {
    cf.print();
  }
  std::cout << "preset_anchors.size():" << preset_anchors.size() <<std::endl;
  for(auto cf:preset_anchors)
  {
    cf.print();
  }
  #endif

  // save as x1,y1,x2,y2
  if (dense_anchor) {
    assert(stride % 2 == 0);
    auto num = preset_anchors.size();
    for (size_t i = 0; i < num; ++i) {
      CRect2f anchor = preset_anchors[i];
      preset_anchors.push_back(CRect2f(anchor[0]+int(stride/2),
            anchor[1]+int(stride/2),
            anchor[2]+int(stride/2),
            anchor[3]+int(stride/2)));
    }
  }

  anchor_stride = stride;

  anchor_num = (int)preset_anchors.size();
  return anchor_num;
}

int AnchorGenerator::FilterAnchor(CProcessOutputTensor *cls, CProcessOutputTensor *reg, CProcessOutputTensor *pts, std::vector<Anchor>& result)
{
  int pts_length = 0;

  const std::vector<int>& pts_shape = pts->shape();
  const std::vector<int>& cls_shape = cls->shape();
  const std::vector<int>& reg_shape = reg->shape();

  pts_length = pts_shape[pts_shape.size() - 3]/anchor_num/2;
  int w = cls_shape[cls_shape.size() - 1];
  int h = cls_shape[cls_shape.size() - 2];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int id = i * w + j;
      for (int a = 0; a < anchor_num; ++a)
      {
        if ( cls->value_at(0, anchor_num + a, i, j) >= cls_threshold) {
          CRect2f box(j * anchor_stride + preset_anchors[a][0],
              i * anchor_stride + preset_anchors[a][1],
              j * anchor_stride + preset_anchors[a][2],
              i * anchor_stride + preset_anchors[a][3]);
          CRect2f delta(reg->value_at(0, a*4+0, i, j),
              reg->value_at(0, a*4+1, i, j),
              reg->value_at(0, a*4+2, i, j),
              reg->value_at(0, a*4+3, i, j));

          Anchor res;
          res.anchor = cv::Rect_< float >(box[0], box[1], box[2], box[3]);
          bbox_pred(box, delta, res.finalbox);
          res.score = cls->value_at(0, anchor_num + a, i, j);
          res.center = cv::Point(j,i);

          if (1) {
            std::vector<cv::Point2f> pts_delta(pts_length);
            for (int p = 0; p < pts_length; ++p) {
              pts_delta[p].x = pts->value_at(0, a*pts_length*2+p*2, i, j);
              pts_delta[p].y = pts->value_at(0, a*pts_length*2+p*2+1, i, j);
            }
            landmark_pred(box, pts_delta, res.pts);
            //                        printf("landmark_pred\n");
          }
          result.push_back(res);
        }
      }
    }
  }
  return 0;
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5f * (w - 1);
  float y_ctr = anchor[1] + 0.5f * (h - 1);

  ratio_anchors.clear();
  float sz = w * h;
  for (size_t s = 0; s < ratios.size(); ++s) {
    float r = ratios[s];
    float size_ratios = sz / r;
    float ws = std::sqrt(size_ratios);
    float hs = ws * r;
    ratio_anchors.push_back(CRect2f(x_ctr - 0.5f * (ws - 1),
          y_ctr - 0.5f * (hs - 1),
          x_ctr + 0.5f * (ws - 1),
          y_ctr + 0.5f * (hs - 1)));
  }
}

void AnchorGenerator::_scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors) {
  scale_anchors.clear();
  for (size_t a = 0; a < ratio_anchor.size(); ++a) {
    CRect2f anchor = ratio_anchor[a];
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5f * (w - 1);
    float y_ctr = anchor[1] + 0.5f * (h - 1);

    for (size_t s = 0; s < scales.size(); ++s) {
      float ws = w * scales[s];
      float hs = h * scales[s];
      scale_anchors.push_back(CRect2f(x_ctr - 0.5f * (ws - 1),
            y_ctr - 0.5f * (hs - 1),
            x_ctr + 0.5f * (ws - 1),
            y_ctr + 0.5f * (hs - 1)));
    }
  }

}

void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& box) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5f * (w - 1);
  float y_ctr = anchor[1] + 0.5f * (h - 1);

  float dx = delta[0];
  float dy = delta[1];
  float dw = delta[2];
  float dh = delta[3];

  float pred_ctr_x = dx * w + x_ctr;
  float pred_ctr_y = dy * h + y_ctr;
  float pred_w = std::exp(dw) * w;
  float pred_h = std::exp(dh) * h;

  box = cv::Rect_< float >(pred_ctr_x - 0.5f * (pred_w - 1.0f),
      pred_ctr_y - 0.5f * (pred_h - 1.0f),
      pred_ctr_x + 0.5f * (pred_w - 1.0f),
      pred_ctr_y + 0.5f * (pred_h - 1.0f));
}

void AnchorGenerator::landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5f * (w - 1);
  float y_ctr = anchor[1] + 0.5f * (h - 1);

  pts.resize(delta.size());
  for (size_t i = 0; i < delta.size(); ++i) {
    pts[i].x = delta[i].x*w + x_ctr;
    pts[i].y = delta[i].y*h + y_ctr;
  }
}


