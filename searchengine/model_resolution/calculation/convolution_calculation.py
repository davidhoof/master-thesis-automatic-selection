class ConvolutionCalculation:

    def __init__(self, operators=["Conv", "ConvTranspose", "ConvInteger", "QLinearConv"]):
        self.__operators = operators

    def get_operators(self):
        return self.__operators

    @staticmethod
    def __check_strides(attribute) -> bool:
        """
        Returns True if strides in attribute affects output size
        :param attribute: given attribute to check
        :return: bool
        """
        return all(i > 0 for i in attribute.ints) and attribute.name == "strides"

    @staticmethod
    def __check_padding(attribute) -> bool:
        """
        Returns True if padding in attribute affects output size
        :param attribute: given attribute to check
        :return: bool
        """
        return all(i > 0 for i in attribute.ints) and attribute.name == "pads"

    @staticmethod
    def __check_kernel_size(attribute) -> bool:
        """
        Returns True if kernel size in attribute affects output size
        :param attribute: given attribute to check
        :return: bool
        """
        return all(i > 0 for i in attribute.ints) and attribute.name == "kernel_shape"

    def filter(self, node) -> bool:
        """
        Filter method to reduce to only necessary operations
        :param node: node to check
        :return: bool
        """
        return (node.op_type in self.__operators) and any(
            [self.__check_strides(attribute) or self.__check_padding(attribute) or self.__check_kernel_size(attribute)
             for
             attribute in node.attribute])

    def __find_attributes(self, node):
        """
        Returns the padding, kernel size and strides values from the attribute
        :param node: node, which to extract the attribute values from
        :return: (padding_width_top, padding_width_bottom, strides_width, kernel_width,
               padding_height_top, padding_height_bottom, strides_height, kernel_height)
        """
        attributes = list(map(list, zip(*[[attribute.name, attribute.ints] for attribute in node.attribute])))

        padding_width_top, padding_width_bottom, padding_height_top, padding_height_bottom = self.__extract_pads(
            attributes)

        strides_width, strides_height = self.__extract_strides(attributes)
        kernel_width, kernel_height = self.__extract_kernel(attributes)

        return padding_width_top, padding_width_bottom, strides_width, kernel_width, \
               padding_height_top, padding_height_bottom, strides_height, kernel_height

    @staticmethod
    def __extract_pads(attributes) -> tuple:
        """
        Extract the padding from the attributes
        :param attributes: attributes to check
        :return: (padding_width_top, padding_width_bottom, padding_height_top, padding_height_bottom)
        """

        if "pads" not in attributes[0]:
            return 0, 0, 0, 0
        return attributes[1][attributes[0].index("pads")][0], attributes[1][attributes[0].index("pads")][1], \
               attributes[1][attributes[0].index("pads")][2], attributes[1][attributes[0].index("pads")][3]

    @staticmethod
    def __extract_strides(attributes) -> tuple:
        """
        Extract the strides from the attributes
        :param attributes: attributes to check
        :return: (strides_width, strides_height)
        """
        return attributes[1][attributes[0].index("strides")][0], attributes[1][attributes[0].index("strides")][1]

    @staticmethod
    def __extract_kernel(attributes) -> tuple:
        """
        Extract the kernel shape from the attributes
        :param attributes: attributes to check
        :return: (kernel_width, kernel_height)
        """
        return attributes[1][attributes[0].index("kernel_shape")][0], \
               attributes[1][attributes[0].index("kernel_shape")][1]

    def calculate_output_height(self, input_height, node):
        """
        Calculates the output height with the formula:
            (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
        :param input_height: height of the input
        :param node: node to calculate output height from
        :return: float
        """
        _, _, _, _, padding_height_top, padding_height_bottom, strides_height, kernel_height = self.__find_attributes(
            node)
        return (input_height + padding_height_top + padding_height_bottom - kernel_height) / strides_height + 1

    def calculate_output_width(self, input_width, node):
        """
        Calculates the output width with the formula:
            (Output width + padding width right + padding width left - kernel width) / (stride width) + 1
        :param input_width: width of the input
        :param node: node to calculate output height from
        :return: float
        """
        padding_width_top, padding_width_bottom, strides_width, kernel_width, _, _, _, _ = self.__find_attributes(node)
        return (input_width + padding_width_top + padding_width_bottom - kernel_width) / strides_width + 1

    def calculate_input_height(self, output_height, node):
        """
        Calculates the input height with the converted formula:
             -padding_height_bottom + kernel_height + strides_height * output_height - strides_height - padding_height_top
        :param output_height: height of the output
        :param node: node to calculate input height from
        :return: float
        """
        _, _, _, _, padding_height_top, padding_height_bottom, strides_height, kernel_height = self.__find_attributes(
            node)
        return -padding_height_bottom + kernel_height + strides_height * output_height - strides_height - padding_height_top

    def calculate_input_width(self, output_width, node):
        """
        Calculates the input width with the converted formula:
             -padding_width_bottom + kernel_width + strides_width * output_width - strides_width - padding_width_top
        :param output_width: width of the output
        :param node: node to calculate input width from
        :return: float
        """
        padding_width_top, padding_width_bottom, strides_width, kernel_width, _, _, _, _ = self.__find_attributes(node)
        return -padding_width_bottom + kernel_width + strides_width * output_width - strides_width - padding_width_top
