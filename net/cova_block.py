import torch
import torch.nn as nn

class CovaBlock(nn.Module):
    def __init__(self):
        super(CovaBlock, self).__init__()

    # calculate the covariance matrix 
    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h * w * B - 1)
            CovaMatrix_list.append(covariance_matrix)
        return CovaMatrix_list

    # calculate the similarity  
    def cal_similarity(self, input, CovaMatrix_list):
        B, C, h, w = input.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).to(query_sam.device)

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.view(1, -1)) # Changed from unsqueeze(0) to view(1, -1) to match ref often

        # Reference uses: Cova_Sim = torch.cat(Cova_Sim, 0)
        # In ref: Cova_Sim.append(mea_sim.view(1, -1)) -> then cat(0)
        Cova_Sim = torch.cat(Cova_Sim, 0) 
        
        return Cova_Sim 

    def forward(self, x1, x2):
        # x1: query features
        # x2: list of support features
        CovaMatrix_list = self.cal_covariance(x2)
        Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)

        return Cova_Sim
