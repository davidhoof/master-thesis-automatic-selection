import os
from collections import defaultdict

import networkx as nx
import onnx

from model_resolution.calculation.convolution_calculation import ConvolutionCalculation
from model_resolution.calculation.pool_calculation import PoolCalculation


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
        if isinstance(onnx_model, str) and os.path.exists(onnx_model):
            raise FileNotFoundError

        self.model = onnx_model
        onnx.checker.check_model(self.model)

        # build up inverted indices
        self.node_by_id = dict()
        self.nodes_by_tensor_input = defaultdict(list)
        self.nodes_by_tensor_output = defaultdict(list)

        for node in self.model.graph.node:
            self.node_by_id[id(node)] = node
            for input_tensor in node.input:
                self.nodes_by_tensor_input[input_tensor].append(id(node))
            for output_tensor in node.output:
                self.nodes_by_tensor_output[output_tensor].append(id(node))

        self.calculators = calculators
        self.filter = [calculator.filter for calculator in self.calculators]
        self.output_size = output_size

        self.longest_path = self.calculate_longest_path()

    def get_calculator(self, node):
        """
        Returns calculator for node if a suitable calculator is found
        :param node: node to check
        :return: calculator
        """
        for calculator in self.calculators:
            if node.op_type in calculator.get_operators():
                return calculator
        return None

    def apply_filters(self, node) -> bool:
        """
        Apply filters on node
        :param node: node to apply filters to
        :return: bool
        """
        return any([filter_f(node) for filter_f in self.filter])

    def filter_nodes(self, node_list: list) -> list:
        """
        Filters nodes to keep only needed nodes for the calculation
        :param node_list: list on which the filters are applied to
        :return: list
        """
        return list(filter(self.apply_filters, node_list))

    def get_node_input_ids(self, node: onnx.NodeProto):
        """
        Generator for the input nodes ids based on given node
        :param node:
        """
        for input_tensor in node.input:
            for connected_node_id in self.nodes_by_tensor_output[input_tensor]:
                yield connected_node_id

    def calculate_longest_path(self):
        """
        Calculates the longest path on the graph
        :return: list
        """
        return [self.node_by_id[node_id] for node_id in nx.dag_longest_path(self.build_graph())]

    def build_graph(self):
        """
        Builds a networkx Directed Acyclic Graph with input scale weights. This scale is calculated with given calculators
        to weight the impact on the edge. If no calculator is found the output size attribute is used as weight
        :return: DiGraph
        """
        G = nx.DiGraph()

        for node in reversed(self.model.graph.node):
            for connected_node in self.get_node_input_ids(node):
                if self.get_calculator(node) is not None:
                    G.add_edge(
                        id(node),
                        connected_node,
                        weight=(self.get_calculator(node).calculate_input_width(self.output_size[0], node) *
                                self.get_calculator(node).calculate_input_height(self.output_size[1], node))
                    )
                else:
                    G.add_edge(id(node), connected_node, weight=self.output_size[0] * self.output_size[1])

        return G

    def calculate_minimal_resolution_with_longest_path(self) -> list:
        """
        Calculates the minimal resolution (min input size) on the longest path based on given calculators
        :return: list
        """
        input_size = [0, 0]
        output_size = self.output_size
        for node in self.filter_nodes(self.longest_path):
            input_size = [self.get_calculator(node).calculate_input_width(output_size[0], node),
                          self.get_calculator(node).calculate_input_height(output_size[1], node)]

            output_size = input_size

        return input_size

    def calculate_minimal_resolution_basic(self) -> list:
        """
        Calculates the minimal resolution (min input size) on the simple nodes list based on given calculators
        :return: list
        """
        input_size = [0, 0]
        output_size = self.output_size
        for node in reversed(self.filter_nodes(self.model.graph.node)):
            input_size = [self.get_calculator(node).calculate_input_width(output_size[0], node),
                          self.get_calculator(node).calculate_input_height(output_size[1], node)]

            output_size = input_size

        return input_size
