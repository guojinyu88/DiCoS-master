import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiRelationalGCN(nn.Module):
    def __init__(self, hidden_size, layer_nums, relation_type):
        super(MultiRelationalGCN, self).__init__()

        self.hidden_size = hidden_size
        self.f_rs = nn.ModuleList()
        for i in range(relation_type):
            self.f_rs.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layer_nums = layer_nums
        self.f_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_g = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, slot_node:torch.Tensor, dialogue_node:torch.Tensor, update_current_mm:torch.Tensor, slot_all_connect:torch.Tensor, update_mm:torch.Tensor, slot_domain_connect:torch.Tensor):
        dialogue_node = dialogue_node.unsqueeze(0).repeat(dialogue_node.shape[0],1,1) 
        for i in range(self.layer_nums):
            dialogue_node_current = self.f_s(dialogue_node)
            slot_node_current = self.f_s(slot_node) 
            
            relation_dialogue_node_neighbour = [] 
            relation_slot_node_neighbour = []


            for f_r in self.f_rs:
                relation_dialogue_node_neighbour.append(f_r(dialogue_node))
                relation_slot_node_neighbour.append(f_r(slot_node))


            update_current_mm_d2s = update_current_mm.matmul(relation_dialogue_node_neighbour[0]) / (update_current_mm.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) 
            update_current_mm_s2d = update_current_mm.transpose(1,2).matmul(relation_slot_node_neighbour[0]) / (update_current_mm.transpose(1,2).sum(-1, keepdim=True).expand_as(dialogue_node_current) + 1e-4) 

            slot_all_connect_s2s = slot_all_connect.matmul(relation_slot_node_neighbour[1]) / (slot_all_connect.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) 
            

            update_mm_d2s = update_mm.matmul(relation_dialogue_node_neighbour[2]) / (update_mm.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) 
            update_mm_s2d = update_mm.transpose(1,2).matmul(relation_slot_node_neighbour[2]) / (update_mm.transpose(1,2).sum(-1, keepdim=True).expand_as(dialogue_node_current) + 1e-4) 


            slot_domain_connect_s2s = slot_domain_connect.matmul(relation_slot_node_neighbour[3]) / (slot_domain_connect.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) 


            dialogue_node_current = dialogue_node_current + update_current_mm_s2d + update_mm_s2d 
            slot_node_current = slot_node_current + update_current_mm_d2s + slot_all_connect_s2s + update_mm_d2s + slot_domain_connect_s2s 

            # 门控更新
            slot_gate = F.sigmoid(self.f_g(torch.cat([slot_node_current, slot_node], dim=-1))) 
            slot_node = (F.relu(slot_node_current) * slot_gate) + (slot_node * (1-slot_gate))

            dialogue_gate = F.sigmoid(self.f_g(torch.cat([dialogue_node_current, dialogue_node], dim=-1))) 
            dialogue_node = (F.relu(dialogue_node_current) * dialogue_gate) + (dialogue_node * (1-dialogue_gate)) 

        return slot_node, dialogue_node
