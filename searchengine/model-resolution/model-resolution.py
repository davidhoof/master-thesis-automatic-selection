import os
import onnx
from .calculation.convolution_calculation import ConvolutionCalculation
from .calculation.pool_calculation import PoolCalculation


class ModelResolution:

    def __init__(self,
                 onnx_model="",
                 calculators=[
                     ConvolutionCalculation(),
                     PoolCalculation()
                 ],
                 output_size=[2, 2]
                 ) -> None:

        if onnx_model is "":
            return
        if isinstance(onnx_model, str) and os.path.exists(onnx_model):
            onnx_model = onnx.load(onnx_model)
        if not isinstance(onnx_model, str) and os.path.exists(onnx_model):
            raise FileNotFoundError

        self.model = onnx_model
        onnx.checker.check_model(self.model)

        self.calculators = calculators
        self.filter = [calculator.filter for calculator in self.calculators]
        self.output_size = output_size
        self.minimal_resolution = self.calculate_minimal_resolution()

    def get_calculator(self, node):
        """
        Returns calculator for node if a suitable calculator is found
        :param node: node to check
        :return: calculator
        """
        for calculator in self.calculators:
            if node.op_type in calculator.get_operators():
                return calculator
        raise ModuleNotFoundError

    def apply_filters(self, node) -> bool:
        """
        Apply filters on node
        :param node: node to apply filters to
        :return: bool
        """
        return any([filter_f(node) for filter_f in self.filter])

    def filter_nodes(self) -> list:
        """
        Filters nodes to keep only needed nodes for the calculation
        :return: list
        """
        return list(filter(self.apply_filters, self.model.graph.node))

    def calculate_minimal_resolution(self) -> list:
        """
        Calculates the minimal resolution (min input size) based on given calculators
        :return: list
        """
        input_size = [0, 0]
        output_size = self.output_size
        for node in reversed(self.filter_nodes()):
            input_size = [self.get_calculator(node).calculate_input_width(output_size[0], node),
                          self.get_calculator(node).calculate_input_height(output_size[1], node)]

            output_size = input_size

        return input_size
