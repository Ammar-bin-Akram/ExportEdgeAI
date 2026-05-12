# # import torch
# # import torchvision

# # checkpoint = torch.load(
# #     r"D:\FYP\models\deeplabv3_mobilenet_mango (1).pth",
# #     map_location="cpu",
# #     weights_only=False
# # )

# # # Check if it's a full model or state dict
# # if hasattr(checkpoint, 'eval'):
# #     model = checkpoint
# # else:
# #     model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
# #         weights=None,
# #         num_classes=2
# #     )
# #     model.load_state_dict(checkpoint)

# # model.eval()

# # dummy_input = torch.randn(1, 3, 224, 224)
# # torch.onnx.export(
# #     model,
# #     dummy_input,
# #     r"D:\FYP\models\segmentation.onnx",
# #     opset_version=11,
# #     input_names=["input"],
# #     output_names=["output"]
# # )
# # print("ONNX export done")
# import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf

# # Load ONNX model
# onnx_model = onnx.load(r"D:\FYP\models\segmentation.onnx")

# # Convert to TensorFlow
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph(r"D:\FYP\models\segmentation_tf")

# # Convert to TFLite
# converter = tf.lite.TFLiteConverter.from_saved_model(r"D:\FYP\models\segmentation_tf")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()

# # Save TFLite model
# with open(r"D:\FYP\models\segmentation.tflite", "wb") as f:
#     f.write(tflite_model)

# print("TFLite conversion done")

import onnx2tf

onnx2tf.convert(
    input_onnx_file_path=r"D:\FYP\models\segmentation.onnx",
    output_folder_path=r"D:\FYP\models\segmentation_tf",
    non_verbose=False
)