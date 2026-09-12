# imca_vision_26aim 项目规则

## 项目入口

- 项目说明、依赖、构建和运行方式：先读 `readme.md`。
- 硬件抽象层：`io/`；应用入口：`src/`；算法功能：`tasks/`；测试程序：`tests/`。
- 项目专属 Skill 的唯一源：`.agents/skills/`。现有历史参考资料位于 `.skill/sp-vision-agent/`，迁移前不得假定两者内容一致。
- 团队共享 AI 知识：`docs/ai/`。

## 执行规则

1. 修改前运行 `git status --short`，保留用户已有的暂存、未暂存和未跟踪改动。
2. 先用 `rg` 阅读真实调用链、配置读取和测试，再决定最小改动范围。
3. 代码修改使用正式 CMake 构建和相关测试验证；硬件、相机、CAN、串口、ROS 或 GPU 环境不可用时，明确记录限制。
4. 改动配置、部署前提、命令或行为时，同步检查 `readme.md` 和 `docs/ai/` 中的对应说明。
5. 不自动执行 `sudo`、设备写入、删除 worktree/分支、`git push`、创建 MR 或发布共享知识；这些动作需要用户明确授权。
6. 一个任务只能有一个主要 Writer；并行代码修改必须使用独立 worktree，并在最终版本上重新测试和核对知识。

## 相关知识

按 `docs/ai/knowledge-index.md` 定位模块、文档、Skill 与验证方式。涉及打符回放、YOLO Pose、PnP 或跨局状态时，先读 `.skill/sp-vision-agent/references/buff.md`；涉及 CAN、串口、udev、systemd 或自启动时，先读 `.skill/sp-vision-agent/references/deployment.md`。

## 交付要求

交付时说明实际修改文件、验证命令及结果、未执行的环境相关验证，并保留完整 Git diff 供用户 Review。知识只根据用户接受或合入的最终代码确认。

