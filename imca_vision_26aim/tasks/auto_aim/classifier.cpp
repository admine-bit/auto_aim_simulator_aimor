#include "classifier.hpp"

#include <algorithm>

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
Classifier::Classifier(const std::string & config_path)
{
  // 分类器输入是 32x32 的单通道装甲板图案。
  engine_ = make_infer_engine(
    config_path, "classify_model", "classify_engine_path", 32, 32, /*channels=*/1);
}

void Classifier::classify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  // 后端统一返回 logits，后续 softmax 保持主分支逻辑。
  cv::Mat outputs = engine_->infer(input);

  // softmax
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

void Classifier::ovclassify(Armor & armor)
{
  classify(armor);
}

}  // namespace auto_aim
