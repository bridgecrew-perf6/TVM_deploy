/*!
 * \brief Example code on load and run TVM module.so
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <cstdio>

void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/mod.so");
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  // tvm::runtime::NDArray x= tvm::runtime::NDArray::Empty({32,32}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);
  // uint8_t *p_data = static_cast<uint8_t*>(x->data);
  // for (int i = 0; i < 300; ++i) {
  //   for (int j = 0; j < 300; ++j) {
  //     p_data[i * 2 + j] = 23;
  //   }
  // }
  // // set the right input
  // set_input("x", x);
  // run the code
  run();
  // get the output
  //get_output(0, y);

  // for (int i = 0; i < 2; ++i) {
  //   for (int j = 0; j < 2; ++j) {
  //     ICHECK_EQ(static_cast<float*>(y->data)[i * 2 + j], i * 2 + j + 1);
  //   }
  // }
}



int main(void) {
  DeployGraphExecutor();
  return 0;
}
