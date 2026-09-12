#!/bin/bash
sleep 1

export MVCAM_COMMON_RUNENV=/opt/MVS/lib

export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol

export ALLUSERSPROFILE=/opt/MVS/MVFG
export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH

# 清理现有ROS进程
#echo "清理现有ROS进程..."
#pkill -f ros  # 杀掉所有ROS2进程
#ros2 daemon stop
#ros2 daemon start
# sleep 1


# cd ~/im_vision_25/ && cmake -B build && make -C build/ -j`nproc`

echo -e "\033[32m编译成功！即将关闭当前终端并打开新终端运行程序。\033[0m"
sleep 1

# 关键修复2：添加 & 让新终端在后台启动，避免阻塞当前终端关闭"
    gnome-terminal -- bash -c " 
    # 新终端中重新加载ROS环境（必做，新终端无继承环境）
    cd ~;  # 切换到主目录
    sleep 2

    # 循环运行程序
    while true; do
        cd ~/im_vision_25/ ;
         ./build/standard_mpc;
        echo '程序已退出，2秒后重启...';
        sleep 2;
        
      # 等待当前终端中的程序执行完毕（避免同时创建多个终端）
       wait $!
    done
" 
