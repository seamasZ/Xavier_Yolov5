# Xavier_Yolov5
将Yolov5部署到Xavier开发板上，并在DeepStream前端显示结果

开发板的用户名是ryan  密码也是ryan

在文件“./英伟达DeepStream5.0中Yolov5的部署及使用——Xavier环境.md ”中描述了
①	开发板的刷机
②	模型转化（用.wts文件作为中间文件）
③	deepstream运行

我用以上这种方式成功部署过，但是用.wts文件的模型转化过程（.pth→.engine）只支持经典的网络结构，比如原始的YOLOv5，在修改网络结构后就不能用这种方式进行模型转化。我用的YOLOv5-v6.1官方代码中的export.py,改一下参数就直接输出.engine
 

如何将YOLOv5部署到deepstream，进行目标检测推理？

①	按照“./英伟达DeepStream5.0中Yolov5的部署及使用——Xavier环境.md”的刷机步骤和安装torch和torchvision。刷机包里jetpack自带了很多依赖库。


②	用YOLOv5-v6.1中的export.py把训练好的.pth转化成.engine，把.engine和对应的标签文件labels.txt保存备用。
开发板在刷机之后就带了deepstream和示例工程文件。
在/opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo是YOLOv3的示例工程文件。将这个文件复制出来，魔改其中的代码作为我的工程文件。开发板桌面上的deepstream-yolov5@6.0文件夹即是改代码之后的工程文件。改了其中的nvdsinfer_custom_impl_yolov5文件夹下面的nvdsparsebbox_Yolo.cpp、trt_utils.cpp和trt_utils.h三个文件。将.engine推理引擎和labels.txt两个文件放在deepstream-yolov5@6.0文件下。需要修改参数配置文件deepstream_app_config_yoloV5.txt中的视频源地址、保存方式、显示参数等。各模块和参数的解释参考
https://www.dandelioncloud.cn/article/details/1565338445545697281 其中在[primary-gie]模块中的config-file=config_infer_primary_V5.txt还指向了配置文件config_infer_primary_V5.txt，但是我没有改里面的参数，默认就能用。


③	运行deepstream程序：
配置好摄像头的ip，按照格式写在deepstream_app_config_yoloV5.txt中。在deepstream-yolov5@6.0下打开terminal，输入命令deepstream-app -c ./deepstream_app_config_yoloV5.txt     程序就能run起来了。
这时候可以在terminal里用jtop命令查看资源占用情况。
程序run起来之后，默认是将结果按rtsp协议输出到rtsp://localhost:8554/ds-test   所以可以在同一网络内的其它设备上看到结果。   在同一个局域网下的另一台windows机器上安装软件Potplayer，播放rtsp://开发板的ip:8554/ds-test，即可实时显示结果。
