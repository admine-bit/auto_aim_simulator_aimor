#include "vc/detector/rune_detector.h"
#include "vc/detector/rune_detector_param.h"
#include "vc/feature/rune_center.h"
#include "vc/feature/rune_group.h"
#include "vc/core/debug_tools.h"
#include "vc/core/debug_tools/window_auto_layout.h"

using namespace std;
using namespace cv;

/**
 * @brief 更新特征组
 *
 * @param[in] matched_features 匹配好的特征
 * @param[out] features 特征组
 */
static inline void setFeatures(const std::vector<RuneFeatureComboConst> &matched_features, std::vector<FeatureNode_cptr> &features)
{
    // 更新特征组
    for (const auto &[target, center, fan] : matched_features)
    {
        if (target)
            features.emplace_back(target);
        if (center)
            features.emplace_back(center);
        if (fan)
            features.emplace_back(fan);
    }
}

/**
 * @brief 设置序列组的基础属性
 */
inline static void setBaseProperties(std::vector<FeatureNode_ptr> &groups, const GyroData &gyro_data, int64 record_time)
{
    for (auto &group : groups)
    {
        auto rune_group = RuneGroup::cast(group);
        rune_group->getPoseCache().setGyroData(gyro_data);
        rune_group->setTick(record_time);
    }
}



void RuneDetector::detect(DetectorInput &input, DetectorOutput &output)
{
    observed_rune_center_.reset();
    auto &groups = input.getFeatureNodes();
    auto &input_image = input.getImage();
    auto &tick = input.getTick();
    auto &gyro_data = input.getGyroData();
    auto &color = input.getColor();
    auto &color_thresh = input.getColorThresh();

    if (groups.size() > 1)
        VC_THROW_ERROR("Size of the argument \"groups\" is greater than 1");
    setInputImage(input_image);
    setTick(tick);
    setGyroData(gyro_data);

    vector<FeatureNode_ptr> current_combos{};   // 当前帧的神符组合体
    vector<FeatureNode_cptr> current_features{}; // 当前帧的所有神符特

    // 初始化存储信息
    if (groups.empty())
        groups.emplace_back(RuneGroup::make_feature());

    setBaseProperties(groups, gyro_data, tick);

    auto rune_group = dynamic_pointer_cast<RuneGroup>(groups.front());
    // 二值化处理图像

    binary(input_image, binary_image_, color, color_thresh);
    const Mat & bin = binary_image_;
#if FEATURE_NODE_DEBUG && !defined(NDEBUG)
    WindowAutoLayout::get()->addWindow("Binary Image");
    imshow("Binary Image", bin);
#endif


    vector<RuneFeatureCombo> matched_features{}; // 匹配好的特征

    // 更新函数
    auto updateRuneGroup = [&]() -> bool
    {
        // 尝试获取神符中心的估计位置
        // 尝试查找所有的神符特征
        if (!findFeatures(bin, current_features, matched_features))
            return false;

        // 强制构造中心只有一个轮廓点；真实 R 轮廓至少有 6 点。这里只在当前图像
        // 识别成功的分支缓存原始中心，掉帧历史重投影分支不会写入。
        for (const auto &[target, center, fan] : matched_features)
        {
            (void)target;
            (void)fan;
            auto rune_center = RuneCenter::cast(center);
            if (!rune_center)
                continue;
            const auto &contours = rune_center->getImageCache().getContours();
            if (contours.empty() || !contours.front() || contours.front()->points().size() < 6)
                continue;
            observed_rune_center_ = rune_center->getImageCache().getCenter();
            break;
        }

        // cout <<"神符特征查找成功" << endl;
        // 尝试获取PNP解算数据

        PoseNode runeGroup_to_cam;
        if (!getPnpData(runeGroup_to_cam, rune_group, to_const(matched_features)))
            return false;

        // 更新序列组
        if (!rune_group->update(runeGroup_to_cam, gyro_data, tick))
            return false;

        // 尝试获取所有的神符组合体
        if (!getRunes(current_combos, rune_group, to_const(matched_features), rune_group->getPoseCache().getPoseNodes()[CoordFrame::CAMERA]))
            return false;

        // 更新神符中心估计信息

        return true;
    };

    // 掉帧状态下的更新函数
    auto updateRuneGroupVanish = [&]() -> bool
    {
        // 掉帧状态处理
        if (!rune_group->visibilityProcess(false))
            return false;

        PoseNode runeGroup_to_cam;
        if (!rune_group->getCamPnpDataFromPast(runeGroup_to_cam))
            return false;

        // 更新序列组
        if (!rune_group->update(runeGroup_to_cam, gyro_data, tick))
            return false;

        // 尝试获取所有的神符组合体
        if (!getRunes(current_combos, rune_group, rune_group->getLastFrameFeatures(), rune_group->getPoseCache().getPoseNodes()[CoordFrame::CAMERA]))
            return false;

        // 更新特征组
        setFeatures(rune_group->getLastFrameFeatures(), current_features);

        return true;
    };

    // 是否为掉帧更新
    bool is_vanish_update = false;
    if (!updateRuneGroup())
    {
        // 这里**不能**再 reset observed_rune_center_。
        // updateRuneGroup() 的失败原因全部在姿态域：getRuneDeviation 的 angle not finite /
        // deviation size 不匹配、getPnpData、RuneGroup::update 的 checkPoseDiff（滤波器创新
        // 门限）/ checkExtremePose / errorValueProcess、getRunes。这些都与"上面那个轮廓是不是
        // 真 R"无关——中心是在 findFeatures 成功后才缓存的（轮廓找到 + filterCenter 概率投票
        // 通过 + 轮廓 ≥6 点），三层筛选一个没少。姿态构造失败就把好中心一起丢掉，会让上层
        // 退到 kpt5 回退路径，实测把整个符中心拉偏。
        // 跨帧安全性：该成员已在 detect() 开头无条件 reset，此处不清不会泄漏上一帧的值。
        is_vanish_update = true;
        // 若更新失败，尝试掉帧状态下的更新
        if (!updateRuneGroupVanish())
        {
            is_vanish_update = true;
            // 若掉帧状态下的更新失败，重新构建神符序列组
            groups = {RuneGroup::make_feature()};
            setBaseProperties(groups, gyro_data, tick);
            output.setFeatureNodes(groups);
            output.setValid(false);
            return;
        }
    }
    if (!is_vanish_update)
    {
        // 若更新成功，清空掉帧数量
        rune_group->visibilityProcess(true);
    }
    Mat img_show = DebugTools::get()->getImage();
    if(matched_features.size() > 1)
        rune_group->drawFeature(img_show);

    if (current_combos.empty())
    {
        VC_THROW_ERROR("组合体为空");
    }

    // 匹配
    auto rune_trackers = rune_group->getTrackers();
    if (!match(current_combos, rune_trackers, is_vanish_update))
    {
        rune_trackers.clear();
        match(current_combos, rune_trackers, is_vanish_update);
    }
    rune_group->setTrackers(rune_trackers);

    rune_group->sync(gyro_data, tick);
    
    output.setValid(true);
    output.setFeatureNodes(groups);
}
