import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference, helper, GraphProto


def add_intermediate_output(model: GraphProto, output_name:str) -> GraphProto:
    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = output_name
    model.graph.output.append(intermediate_layer_value_info)

    return onnx.shape_inference.infer_shapes(model)


if __name__ == '__main__':
    # This script will take a Yolov5s onnx export and adjust it to allow exporting an intermediate layer
    existing_onnx_path = "yolov5l.onnx"
    new_onnx_path = "yolov5l_op11_rossac.onnx"

    model = onnx.load_model(existing_onnx_path)

    # 361 works well for yolov5s
    # 603 is the equivalent for yolov5l

    model = add_intermediate_output(model, "603")
    onnx.save_model(model, new_onnx_path)

    onnx.checker.check_model(onnx.load(new_onnx_path), full_check=True)