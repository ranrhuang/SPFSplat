import torch
from jaxtyping import Int
from torch import Tensor


def add_third_context_index(
    indices: Int[Tensor, "*batch 2"], 
    target_indices
) -> Int[Tensor, "*batch 3"]:
    left, right = indices.unbind(dim=-1)
    # print((left + right) // 2, target_indices, torch.isin((left + right) // 2,  target_indices).item())
    if torch.isin((left + right) // 2,  target_indices).item() == False:
        return torch.stack((left, (left + right) // 2, right), dim=-1)
    else:
        return add_more_context_index(indices, 3, target_indices)

def add_more_context_index(
    indices: Int[Tensor, "*batch 2"],
    num_context_views,
    target_indices,
)-> Int[Tensor, "*batch n"]:
    left, right = indices.unbind(dim=-1)
    num_extra_views = num_context_views - 2
    extra_views = []
    
    while len(set(extra_views)) != num_extra_views:
        extra_views = torch.randint(
            left + 1,
            right,
            (num_extra_views,),
        ).tolist()
        extra_views = [x for x in extra_views if x not in target_indices.tolist()]

    # print(indices, extra_views, target_indices)
    return torch.tensor((left, *extra_views, right))