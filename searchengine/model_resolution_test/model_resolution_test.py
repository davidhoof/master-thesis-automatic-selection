import unittest
import onnx

from model_resolution.model_resolution import ModelResolution


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # IMPORTANT: An onnx file has to be provided to run these tests
        onnx_model_path = "G:/Users/David/Downloads/resnet152-v2-7.onnx"

        self.model = onnx.load(onnx_model_path)
        onnx.checker.check_model(self.model)

    def test_something(self):

        mr = ModelResolution(self.model)
        print([node.name for node in mr.longest_path] )
        print(mr.minimal_resolution)

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
