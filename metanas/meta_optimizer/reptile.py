""" REPTILE meta learning algorithm
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""


from collections import OrderedDict,defaultdict
import torch
from utils import utils
import copy
class NAS_Reptile:
    def __init__(self, meta_model,meta_cell, config):
        self.meta_model = meta_model
        self.config = config
        self.meta_cell = meta_cell
        if config.w_meta_optim is None:
            self.w_meta_optim = torch.optim.Adam(
                self.meta_model.weights(), lr=self.config.w_meta_lr
            )

        else:
            self.w_meta_optim = self.config.w_meta_optim
        if meta_cell is None:
            print("meta_cell is empty, not being passed")
            exit(0)
        if config.a_meta_optim is None:
            if meta_model.alphas() is not None:
                print("found alphas, set meta optim")
                self.a_meta_optim = torch.optim.Adam(
                    self.meta_model.alphas(), lr=self.config.a_meta_lr
                )
            else:
                print("-------- no alphas, no meta optim ------")

        else:
            self.a_meta_optim = self.config.a_meta_optim

        self.w_meta_cell_optim = torch.optim.SGD(meta_cell.parameters(), lr = config.w_meta_lr)


        # self.a_meta_cell_optim = torch.optim.SGD(meta_cell.alphas(), lr= config.a_meta_lr)
        
    def step(self, task_infos, meta_cell):

        # Extract infos provided by the task_optimizer
        # k_way = task_infos[0].k_way
        # data_shape = task_infos[0].data_shape

        w_tasks = [task_info.w_task for task_info in task_infos]
        a_tasks = [task_info.a_task for task_info in task_infos]

        w_tasks_power = [task_info.w_task_power for task_info in task_infos]
        a_tasks_power = [task_info.a_task_power for task_info in task_infos]
        
        w_const_bottom = get_const_bottom(w_tasks_power)
        a_const_bottom = get_const_bottom(a_tasks_power)


        


        print("MB log, " ,utils.count_parameters_in_MB(self.meta_model.net))
        self.w_meta_optim.zero_grad()
        self.a_meta_optim.zero_grad()

        self.meta_model.train()

        w_finite_differences = list()
        a_finite_differences = list()

        w_const_list = list()
        a_const_list = list()

        w_cell_finite_difference = list()
        a_cell_finite_difference = list()


        for task_info in task_infos:
            w_finite_differences += [
                get_finite_difference(self.meta_model.named_weights(), task_info.w_task)
            ]
            a_finite_differences += [
                get_finite_difference(self.meta_model.named_alphas(), task_info.a_task)
            ]
        #########################################
        bo_experiment = 1
        cell_start = 1
        #########################################

        if bo_experiment:
            for idx, task_info in enumerate(task_infos):
                w_const_list += [
                    get_const(w_finite_differences[idx],w_const_bottom)
                    ]
                a_const_list += [
                    get_const(a_finite_differences[idx],a_const_bottom)
                ]
        
            for task_idx in range(len(task_infos)):
                
                for layer_name, layer_weight_tensor in self.meta_model.named_weights():
                    const = w_const_list[task_idx][layer_name]
                    if layer_weight_tensor.grad is not None:
                        layer_weight_tensor.grad.data.add_(-(w_finite_differences[task_idx][layer_name]*const)/len(task_infos)) #*w_const_list[task_idx][layer_name])

                
                self.w_meta_optim.step()
                if self.a_meta_optim is not None:
                    self.a_meta_optim.step()

            if cell_start:
               
                

                for idx in range(len(self.meta_model.net.cells)):           #calculate difference
                    meta_model_cell_dict = make_dict(self.meta_model.net.cells[idx].named_parameters())
                    w_cell_finite_difference += [
                        get_finite_difference(meta_cell.named_parameters(), meta_model_cell_dict)
                    ]
                    # a_cell_finite_difference += [
                    #     get_finite_difference(meta_cell.named_weights(), self.meta_model.net.cells[idx].named_weights())
                    # ]

                for idx in range(len(self.meta_model.net.cells)):           #calculate gradients at cell scale
                    for layer_name, layer_weight_tensor in self.meta_cell.named_parameters():
                        if layer_weight_tensor.grad is not None:
                            layer_weight_tensor.grad.data.add_(-w_cell_finite_difference[idx][layer_name]/len(self.meta_model.net.cells))
                
                self.w_meta_cell_optim.step()
                print("cell optimized")
                # if self.a_meta_cell_optim is not None:
                #     self.a_meta_cell_optim.step()


                

        
        else:

            mean_w_task_finitediff = {
                k: get_mean_gradient_from_key(k, w_finite_differences)
                for k in w_tasks[0].keys()
            }

            mean_a_task_finitediff = {
                k: get_mean_gradient_from_key(k, a_finite_differences)
                for k in a_tasks[0].keys()
            }

            for layer_name, layer_weight_tensor in self.meta_model.named_weights():
                if layer_weight_tensor.grad is not None:
                    layer_weight_tensor.grad.data.add_(-mean_w_task_finitediff[layer_name])

            for layer_name, layer_weight_tensor in self.meta_model.named_alphas():
                if layer_weight_tensor.grad is not None:
                    layer_weight_tensor.grad.data.add_(-mean_a_task_finitediff[layer_name])

            self.w_meta_optim.step()
            if self.a_meta_optim is not None:
                self.a_meta_optim.step()

def make_dict(generator_entity):

    dictified = {
        layer_name: copy.deepcopy(layer_weight)
        for layer_name, layer_weight in generator_entity
    }

    return dictified





def get_const_bottom(dict_collections):
    result = defaultdict(float)

    for task_power in dict_collections:
        for layer_name,weight_values in task_power.items():
            result[layer_name] += weight_values

    for layer_name,weight_values in result.items():
        result[layer_name] = torch.sqrt(torch.exp(weight_values))

    return result

def get_const(task_specific_power,const_bottom):
    result = defaultdict(float)

    for layer_name,weight_values in task_specific_power.items():
        result[layer_name] = torch.div(torch.sqrt(torch.exp(task_specific_power[layer_name])),const_bottom[layer_name])

    return result

# some helper functions
def get_finite_difference(meta_weights, task_weights):
    
    for layer_name, layer_weight_tensor in meta_weights:

        if layer_weight_tensor.grad is not None:
            task_weights[layer_name].data.sub_(layer_weight_tensor.data)

    return task_weights  # = task weights - meta weights


def get_mean_gradient_from_key(k, task_gradients):
    return torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
