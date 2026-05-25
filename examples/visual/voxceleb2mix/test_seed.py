import os
 import torch
 import torch.distributed as dist

 def main():
     dist.init_process_group(backend="nccl")
     local_rank = int(os.environ["LOCAL_RANK"])
     torch.cuda.set_device(local_rank)
     device = torch.device(f"cuda:{local_rank}")

     num_visible = torch.cuda.device_count()
     print(f"[rank {local_rank}] Can see {num_visible} GPU(s). "
           f"Active device: cuda:{torch.cuda.current_device()}\n")

     dist.barrier()

     SEED = 42
     if dist.get_rank() == 0:
         torch.random.manual_seed(SEED)

     dist.barrier()

     weight = torch.randn(3, 3, device=device)
     print(f"[rank {local_rank}] Same seed={SEED} → weight[0]: {weight[0].tolist()}\n")

     dist.barrier()

 if __name__ == "__main__":
     main()
