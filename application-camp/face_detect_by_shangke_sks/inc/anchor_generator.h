#ifndef ANCHOR_GENERTOR
#define ANCHOR_GENERTOR

#include <stdio.h>
#include <vector>
#include <iostream>

#include <opencv2/core.hpp>

using namespace std;

class AnchorCfg
{
public:
  vector<float> SCALES;
  vector<float> RATIOS;
  const int BASE_SIZE{0};

  AnchorCfg() {}
  ~AnchorCfg() {}
  AnchorCfg(const vector<float> s, const vector<float> r, int size)
    : BASE_SIZE( size )
  {
    SCALES = s;
    RATIOS = r;
  }
};

class CRect2f
{
public:
  CRect2f(float x1, float y1, float x2, float y2)
  {
    val[0] = x1;
    val[1] = y1;
    val[2] = x2;
    val[3] = y2;
  }

  float &operator[](int i)
  {
    return val[i];
  }

  float operator[](int i) const
  {
    return val[i];
  }

  float val[4];

  void print()
  {
    printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
  }
};

class Anchor
{
public:
  Anchor()
  {
  }

  ~Anchor()
  {
  }

  bool operator<(const Anchor &t) const
  {
    return score < t.score;
  }

  bool operator>(const Anchor &t) const
  {
    return score > t.score;
  }

  float &operator[](int i)
  {
    assert(0 <= i && i <= 4);

    if (i == 0)
      return finalbox.x;
    if (i == 1)
      return finalbox.y;
    if (i == 2)
      return finalbox.width;
    // if (i == 3)
      return finalbox.height;
  }

  float operator[](int i) const
  {
    assert(0 <= i && i <= 4);

    if (i == 0)
      return finalbox.x;
    if (i == 1)
      return finalbox.y;
    if (i == 2)
      return finalbox.width;
    // if (i == 3)
      return finalbox.height;
  }

  cv::Rect_<float> anchor;      // x1,y1,x2,y2
  float reg[4];                    // offset reg
  cv::Point center;             // anchor feat center
  float score;                     // cls score
  std::vector<cv::Point2f> pts; // pred pts

  cv::Rect_<float> finalbox;    // final box res

  void print()
  {
    printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, score);
    printf("landmarks ");
    for (size_t i = 0; i < pts.size(); ++i) {
        printf("%f %f, ", pts[i].x, pts[i].y);
    }
    printf("\n");
  }
};

class CProcessOutputTensor;

class AnchorGenerator
{
public:
  AnchorGenerator();
  ~AnchorGenerator();

  // init different anchors
  int Init(int stride, const AnchorCfg &cfg, bool dense_anchor);

  // anchor plane
  int Generate(int fwidth, int fheight, int stride, float step, std::vector<int> &size, std::vector<float> &ratio, bool dense_anchor);

  // filter anchors and return valid anchors
  int FilterAnchor(CProcessOutputTensor *cls, CProcessOutputTensor *reg, CProcessOutputTensor *pts, std::vector<Anchor> &result);

private:
  void _ratio_enum(const CRect2f &anchor, const std::vector<float> &ratios, std::vector<CRect2f> &ratio_anchors);

  void _scale_enum(const std::vector<CRect2f> &ratio_anchor, const std::vector<float> &scales, std::vector<CRect2f> &scale_anchors);

  void bbox_pred(const CRect2f &anchor, const CRect2f &delta, cv::Rect_<float> &box);

  void landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f> &delta, std::vector<cv::Point2f> &pts);

  std::vector<std::vector<Anchor>> anchor_planes; // corrspont to channels

  std::vector<int> anchor_size;
  std::vector<float> anchor_ratio;
  float anchor_step; // scale step
  int anchor_stride; // anchor tile stride
  int feature_w;     // feature map width
  int feature_h;     // feature map height

  std::vector<CRect2f> preset_anchors;
  int anchor_num; // anchor type num
  float cls_threshold;
};

#endif // ANCHOR_GENERTOR