'''
Now I hate bosch
'''
import torch 


class layer_Reptile:
    def __init__(self,meta_data_layer, config):
        self.meta_cell = meta_data_layer
        self.config = config

        if config.w_meta_optim is None:
            self.w_meta_optim = torch.optim.Adam(
                self.meta_model.weights(), lr=self.config.w_meta_lr
            )

        else:
            self.w_meta_optim = self.config.w_meta_optim

        if config.a_meta_optim is None:
            if meta_data_layer.alphas() is not None:
                print("found alphas, set meta optim")
                self.a_meta_optim = torch.optim.Adam(
                    self.meta_model.alphas(), lr=self.config.a_meta_lr
                )
            else:
                print("-------- no alphas, no meta optim ------")

        else:
            self.a_meta_optim = self.config.a_meta_optim

    def step(self, task_info):

        # just use single-task info for the steps. 

        w_task = task_info.w_task
        a_task = task_info.a_task

        self.w_meta_optim.zero_grad()
        self.a_meta_optim.zero_grad()

        self.meta_cell.train()

        

        for layer_name , layer_weight_tensor in self.meta_cell

# some helper functions
def get_finite_difference(meta_weights, task_weights):

    for layer_name, layer_weight_tensor in meta_weights:

        if layer_weight_tensor.grad is not None:
            task_weights[layer_name].data.sub_(layer_weight_tensor.data)

    return task_weights  # = task weights - meta weights


def get_mean_gradient_from_key(k, task_gradients):
    return torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)