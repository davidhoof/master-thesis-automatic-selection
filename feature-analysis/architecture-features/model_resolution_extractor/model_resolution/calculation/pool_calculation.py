from model_resolution.calculation.convolution_calculation import ConvolutionCalculation


class PoolCalculation(ConvolutionCalculation):
    def __init__(self, operators=["AveragePool", "LpPool", "MaxPool", "MaxRoiPool"]):
        super().__init__(operators)
        self.__operators = operators

    def filter(self, node) -> bool:
        return node.op_type in self.__operators
