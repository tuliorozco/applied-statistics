
import os
import torch
import torch.distributed as dist

def setup_distributed_training():
    """Configura el entorno de entrenamiento distribuido"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}')