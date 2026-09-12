#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "armor.hpp"
#include "inference/infer_engine.hpp"

namespace auto_aim
{
class Classifier
{
public:
  explicit Classifier(const std::string & config_path);

  void classify(Armor & armor);

  void ovclassify(Armor & armor);

private:
  std::unique_ptr<InferEngine> engine_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP
