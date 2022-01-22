1 首先需要编译好TVM  得到libtvm.so 和libtvm.runtime.so这两个库文件


2 准备要推理的模型 得到module.so文件
complie.py  可以编译tflite模型

3 进行部署，加载module.so，进行推理
python_deploy.py
cpp_deploy.cc





build :

```bazel build main --cxxopt='-std=c++14'```


cd -r ./lib ./bazel-bin
./bazel-bin/main

