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

import torch


class NAS_Reptile_layer:
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

    def step(self, task_info):

        # Extract infos provided by the task_optimizer
        # k_way = task_infos[0].k_way
        # data_shape = task_infos[0].data_shape

        w_tasks = task_info.w_task 
        a_tasks = task_info.a_task
        
        self.w_meta_optim.zero_grad()
        self.a_meta_optim.zero_grad()

        self.meta_model.train()
        self.meta_cell.train()

        w_cell_diff = get_cell_finite_difference(self.meta_cell.named_weights(),task_info.w_task)
        a_cell_diff = get_cell_finite_difference(self.meta_cell.named_alphas(),task_info.a_task)
        
        
        

        mean_w_task_finitediff = {
            k: get_mean_gradient_from_key(k, w_cell_diff)
            for k in w_tasks[0].keys()
        }
        mean_a_task_finitediff = {
            k: get_mean_gradient_from_key(k, a_cell_diff)
            for k in a_tasks[0].keys()
        }
        #이 부분 작동 안할 가능성
        
        for layer_name, layer_weight_tensor in w_tasks:
            if layer_weight_tensor is not None:
                self.meta_cell.edges.grad.data.add_(-mean_w_task_finitediff[layer_name])
        
        for layer_name, layer_weight_tensor in a_tasks:
            if layer_weight_tensor is not None:
                self.meta_cell.edges.grad.data.add_(-mean_a_task_finitediff[layer_name])    

                

        self.w_meta_optim.step()
        if self.a_meta_optim is not None:
            self.a_meta_optim.step()
            
        #이 뒤 바깥에서 해당 셀로 다시 모델 생성 및 load_state 해줄것 


# some helper functions
def get_cell_finite_difference(meta_weights, task_weights):
    
    for layer_name,layer_weight_tensor in task_weights:
        
        if layer_weight_tensor.grad is not None:
            meta_weights.data.sub_(layer_weight_tensor.data)

    

    return meta_weights  # = task weights - meta weights


def get_mean_gradient_from_key(k, task_gradients):
    return torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
