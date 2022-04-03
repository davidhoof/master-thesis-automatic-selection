import unittest

import onnx

from model_resolution.model_resolution import ModelResolution


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # IMPORTANT: An onnx file has to be provided to run these tests
        # onnx_model_path = "G:/Users/David/Downloads/mnist-7 (4).onnx"
        # onnx_model_path = "G:/Users/David/Downloads/resnet152-v2-7.onnx"
        onnx_model_path = "G:/Users/David/Downloads/bvlcalexnet-12.onnx"

        self.model = onnx.load(onnx_model_path)
        onnx.checker.check_model(self.model)

        # for node in self.model.graph.node:
        #     for attribute in node.attribute:
        #         if (attribute.name == "strides"):
        #             print(attribute.ints)

    def test_something(self):
        mr = ModelResolution(self.model, output_size=[2, 2], use_longest_path=True)
        print(list(reversed([node.output for node in mr.longest_path])))
        print(mr.minimal_resolution)

        print(mr.check_minimal_resolution_with_longest_path())

        self.assertEqual(False,
                         "resnetv27_stage4_conv3_fwd" in list(reversed([node.output for node in mr.longest_path])))


if __name__ == '__main__':
    unittest.main()
