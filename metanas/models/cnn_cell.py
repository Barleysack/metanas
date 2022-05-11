import torch
import torch.nn as nn
import ops 


class SearchCell(nn.Module):
    """Cell for searchs
    Each edge is mixed and continuous relaxed.

    Attributes:
        dag: List of lists where the out list corresponds to intermediate nodes in a cell. The inner
            list contains the mixed operations for each input node of an intermediate node (i.e.
            dag[i][j] calculates the outputs of the i-th intermediate node for its j-th input).
        preproc0: Preprocessing operation for the s0 input
        preproc1: Preprocessing operation for the s1 input
    """

    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, PRIMITIVES):
        """
        Args:
            n_nodes: Number of intermediate n_nodes. The output of the cell is calculated by
                concatenating the outputs of all intermediate nodes in the cell.
            C_pp (int): C_out[k-2]
            C_p (int) : C_out[k-1]
            C (int)   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(C, stride, PRIMITIVES)
                self.dag[i].append(op)

    def forward(
        self, s0, s1, w_dag, w_input=None, w_pw=None, alpha_prune_threshold=0.0
    ):
        """Forward pass through the cell

        Args:
            s0: Output of the k-2 cell
            s1: Output of the k-1 cell
            w_dag: MixedOp weights ("alphas") (e.g. for n nodes and k primitive operations should be
                a list of length `n` of parameters where the n-th parameter has shape
                :math:`(n+2)xk = (number of inputs to the node) x (primitive operations)`)
            w_input: Distribution over inputs for each node (e.g. for n nodes should be a list of
                parameters of length `n`, where the n-th parameter has shape
                :math:`(n+2) = (number of inputs nodes)`).
            w_pw: weights on pairwise inputs for soft-pruning of inputs (e.g. for n nodes should be
                a list of parameters of length `n`, where the n-th parameter has shape
                :math:`(n+2) choose 2 = (number of combinations of two input nodes)`)
            alpha_prune_threshold:

        Returns:
            The output tensor of the cell
        """

        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]

        if w_input is not None:  # hierarchical alphas #####deprecated

            # iterate over nodes in cell
            for edges, w_node_ops, w_node_in in zip(self.dag, w_dag, w_input):
                s_cur = 0.0

                # iterate over inputs of node
                for i, (state_in, w_ops, w_in) in enumerate(
                    zip(states, w_node_ops, w_node_in)
                ):
                    if w_in > 0:
                        s_cur = s_cur + w_in * edges[i](state_in, w_ops)

                # equivalent but harder to read:
                # s_cur2 = sum(w2 * edges[i](s, w)
                #             for i, (s, w, w2) in enumerate(zip(states, w_node_ops, w_node_in)))
                # assert torch.allclose(s_cur2, s_cur)

                states.append(s_cur)

        elif w_pw is not None:  # pairwise alphas

            # iterate over nodes in cell
            for edges, w_node_ops, w_node_pw in zip(self.dag, w_dag, w_pw):
                pairwise_inputs = list()  # pairwise inputs
                unariy_inputs = list()  # unariy/single inputs

                # iterate over inputs per node
                for i, (state_in, w_ops) in enumerate(zip(states, w_node_ops)):

                    input_cur = edges[i](
                        state_in, w_ops, alpha_prune_threshold=alpha_prune_threshold
                    )
                    unariy_inputs.append(input_cur)

                # build pairwise sums
                for input_1 in range(len(unariy_inputs)):
                    for input_2 in range(input_1 + 1, len(unariy_inputs)):
                        pairwise_inputs.append(
                            unariy_inputs[input_1] + unariy_inputs[input_2]
                        )

                assert len(pairwise_inputs) == len(
                    w_node_pw
                ), "error: pairwise alpha length does not match pairwise terms length"

                s_cur = 0.0
                for i, sum_pw in enumerate(
                    pairwise_inputs
                ):  # weight pairwise sums by pw alpha
                    if w_node_pw[i] > alpha_prune_threshold:
                        s_cur = s_cur + sum_pw * w_node_pw[i]

                states.append(s_cur)

        else:  # regular darts

            for edges, w_list in zip(self.dag, w_dag):
                s_cur = sum(
                    edges[i](s, w, alpha_prune_threshold=alpha_prune_threshold)
                    for i, (s, w) in enumerate(zip(states, w_list))
                )

                states.append(s_cur)

        s_out = torch.cat(
            states[2:], dim=1
        )  # concatenate all intermediate nodes except inputs
        return s_out