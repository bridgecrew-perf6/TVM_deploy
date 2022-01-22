import tvm
from tvm.contrib import  graph_executor, download
import numpy as np
from PIL import Image
import os



def get_real_image(im_height, im_width):
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download.download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


if __name__ == "__main__":
    input_dict={"normalized_input_image_tensor": (1, 300, 300, 3)}
    output_counts = 4
    inputs = {}
    for input_name in input_dict:
        input_shape = input_dict[input_name]
        inputs[input_name] = get_real_image(input_shape[1], input_shape[2])
    
    lib_path = "./lib/mod.so"
    lib = tvm.runtime.load_module(lib_path)
    module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(**inputs)
    module.run()
    out = [module.get_output(i).numpy() for i in range(output_counts)]
    
    print(out[0].shape)  #coordinate
    print(out[1])
    print(out[2])
    print(out[3])    
    