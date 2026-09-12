# 代码—知识—验证索引

该索引只表达“可能受影响”的关联，不直接断言文档正确。代码变更后，应检查对应文档、命令和测试是否仍然存在。

| 代码范围 | 相关知识/入口 | 验证方式 |
| --- | --- | --- |
| `src/` | `readme.md` 的运行与应用入口说明 | 配置项目后构建受影响目标；必要时运行对应 demo |
| `io/` | `.skill/sp-vision-agent/references/deployment.md`；`readme.md` 的设备与部署说明 | 静态检查配置；设备可用时验证 CAN/串口/相机 |
| `tasks/auto_aim/` | `readme.md` 的自瞄架构；`.skill/sp-vision-agent/SKILL.md` | 构建 `auto_aim_test` 或受影响目标；运行可用的离线测试 |
| `tasks/auto_buff/` | `.skill/sp-vision-agent/references/buff.md`；打符相关说明 | 构建 `auto_buff_test`；有录制数据时运行离线回放 |
| `tasks/omniperception/` | `readme.md` 的哨兵/全向感知说明 | 在 ROS 依赖可用时构建相关目标 |
| `tests/` | 各测试源文件及其 CMake 目标 | 只运行受影响测试，记录代码版本和环境限制 |
| `cmake/`、顶层 `CMakeLists.txt` | `readme.md` 编译依赖与后端说明 | `cmake -B build` 或等价配置检查 |

## 复核字段

每次知识核对至少记录：代码版本（`base`、`agent`、`final`）、检查日期、执行的命令、结果、未执行项及原因。

