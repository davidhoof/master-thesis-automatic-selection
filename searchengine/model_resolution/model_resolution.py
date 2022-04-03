import os
from collections import defaultdict

import networkx as nx
import numpy as np
import onnx

from model_resolution.calculation.convolution_calculation import ConvolutionCalculation
from model_resolution.calculation.pool_calculation import PoolCalculation


class ModelResolution:

    def __init__(self,
                 model="",
                 calculators=[
                     ConvolutionCalculation(),
                     PoolCalculation()
                 ],
                 max_input_size=[4096, 4096],
                 use_longest_path=True
                 ) -> None:

        if model is "":
            return
        if isinstance(model, str) and os.path.exists(model):
            model = onnx.load(model)
        if isinstance(model, str) and os.path.exists(model):
            raise FileNotFoundError

        self.model = model
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
        self.max_input_size = max_input_size

        if use_longest_path:
            self.longest_path = self.__calculate_longest_path()
            self.minimal_resolution = self.calculate_minimal_resolution_with_longest_path()
        else:
            self.minimal_resolution = self.calculate_minimal_resolution_basic()

    def __get_calculator(self, node: onnx.NodeProto):
        """
        Returns calculator for node if a suitable calculator is found
        :param node: node to check
        :return: calculator
        """
        for calculator in self.calculators:
            if node.op_type in calculator.get_operators():
                return calculator
        return None

    def __apply_filters(self, node: onnx.NodeProto) -> bool:
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
        return list(filter(self.__apply_filters, node_list))

    def __get_node_input_ids(self, node: onnx.NodeProto):
        """
        Generator for the input nodes ids based on given node
        :param node:
        """
        for input_tensor in node.input:
            for connected_node_id in self.nodes_by_tensor_output[input_tensor]:
                yield connected_node_id

    def __calculate_longest_path(self) -> list:
        """
        Calculates the longest path on the graph
        :return: list
        """
        return [self.node_by_id[node_id] for node_id in nx.dag_longest_path(self.__build_graph())]

    def __build_graph(self) -> nx.DiGraph:
        """
        Builds a networkx Directed Acyclic Graph with output scale weights. This scale is calculated with given calculators
        to weight the impact on the edge. If no calculator is found, (0) is used as weight
        :return: DiGraph
        """
        G = nx.DiGraph()

        for node in self.model.graph.node:
            for connected_node in self.__get_node_input_ids(node):
                if self.__get_calculator(node) is not None:
                    G.add_edge(
                        id(node),
                        connected_node,
                        # calculating the weight as proportion scale: max_input_size/calculated output size
                        weight=(
                                (self.max_input_size[0] /
                                 self.__get_calculator(node).calculate_output_width(self.max_input_size[0], node))
                                *
                                (self.max_input_size[1] /
                                 self.__get_calculator(node).calculate_output_height(self.max_input_size[1], node))
                        )
                    )

                else:
                    G.add_edge(id(node), connected_node, weight=0)

        return G

    def calculate_minimal_resolution_with_longest_path(self) -> list:
        """
        Calculates the minimal resolution (min input size) on the longest path based on given calculators
        :return: list
        """
        input_size = np.array(self.max_input_size, dtype=int)
        output_size = np.array([0, 0], dtype=int)
        for node in self.filter_nodes(self.longest_path):
            output_size = np.array([self.__get_calculator(node).calculate_output_width(input_size[0], node),
                                    self.__get_calculator(node).calculate_output_height(input_size[1], node)])

            input_size = output_size

        return list(np.floor(self.max_input_size / output_size).astype(int))

    def calculate_minimal_resolution_basic(self) -> list:
        """
        Calculates the minimal resolution (min input size) on the simple nodes list based on given calculators
        :return: list
        """
        input_size = np.array(self.max_input_size, dtype=int)
        output_size = np.array([0, 0], dtype=int)
        for node in self.filter_nodes(self.model.graph.node):
            output_size = np.array([self.__get_calculator(node).calculate_output_width(input_size[0], node),
                                    self.__get_calculator(node).calculate_output_height(input_size[1], node)])

            input_size = output_size

        return list(np.floor(self.max_input_size / output_size).astype(int))
