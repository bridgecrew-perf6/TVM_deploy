import tvm
from tvm import relay
import tflite.Model

def get_tflite_model(tflite_model_path, inputs_dict, dtype):
    with open(tflite_model_path, "rb") as f:
        tflite_model_buffer = f.read()
    try:
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buffer, 0)
    except AttributeError:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buffer, 0)
    shape_dict = {}
    dtype_dict = {}
    for input in inputs_dict:
        input_shape = inputs_dict[input]
        shape_dict[input] = input_shape
        dtype_dict[input] = dtype

    return relay.frontend.from_tflite(
        tflite_model,
        shape_dict=shape_dict,
        dtype_dict=dtype_dict,
    )


if __name__ == "__main__":
    model_path = "./model/detect.tflite"  #ssd_mobilenet_v1_1.0_
    input_dict={"normalized_input_image_tensor": (1, 300, 300, 3)}
    mod, params = get_tflite_model(model_path, input_dict, "uint8")
    relay.backend.te_compiler.get().clear()
    with tvm.target.Target("llvm"):
        lib = relay.build(mod, params=params)

    lib_path = "./lib/mod.so"
    lib.export_library(lib_path)
   