import torch
import torch.distributed as dist
import pickle


def _is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_info():
    if _is_dist_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def dist_all_gather(obj):
    # wrapper for dist.all_gather_object
    if not _is_dist_initialized() or dist.get_world_size() == 1:
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered

def dist_broadcast_object(obj, src=0):
    """
    Broadcast arbitrary python object from src to everyone.
    Uses broadcast_object_list which requires a list input.
    """
    if not _is_dist_initialized() or dist.get_world_size() == 1:
        return obj
    obj_list = [obj]
    # Needs a list pre-sized on non-src ranks for some pytorch versions, ensure length 1 everywhere.
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def dist_all_gather_object_fallback(obj):
    """
    Gather arbitrary Python objects from all ranks into a list.
    Uses dist.all_gather_object when available; otherwise falls back
    to a pickle->byte-tensor all_gather trick.
    Returns: list_of_objects (length world_size)
    """
    if not _is_dist_avail_and_initialized():
        return [obj]

    world_size = dist.get_world_size()

    # preferred: use the high-level API if present
    if hasattr(dist, "all_gather_object"):
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, obj)
        return gathered

    # fallback: pickle -> byte tensor -> all_gather
    pick = pickle.dumps(obj)
    byte_tensor = torch.ByteTensor(list(pick)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # gather lengths first
    local_len = torch.LongTensor([byte_tensor.numel()]).to(byte_tensor.device)
    lengths = [torch.LongTensor([0]).to(byte_tensor.device) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    lengths = [int(l.item()) for l in lengths]
    max_len = max(lengths)

    # pad to max_len
    if byte_tensor.numel() < max_len:
        padding = torch.ByteTensor(max_len - byte_tensor.numel()).to(byte_tensor.device)
        byte_tensor = torch.cat([byte_tensor, padding], dim=0)

    # gather all byte tensors
    gather_list = [torch.empty(max_len, dtype=torch.uint8, device=byte_tensor.device) for _ in range(world_size)]
    dist.all_gather(gather_list, byte_tensor)

    results = []
    for i, (g, ln) in enumerate(zip(gather_list, lengths)):
        data = bytes(g[:ln].tolist())
        results.append(pickle.loads(data))
    return results

def dist_broadcast_object_fallback(obj, src=0):
    """
    Broadcast a Python object from src to all ranks.
    Uses broadcast_object_list if available; otherwise uses pickle->byte-tensor broadcast.
    Returns the object on all ranks.
    """
    if not _is_dist_avail_and_initialized():
        return obj

    if hasattr(dist, "broadcast_object_list"):
        obj_list = [None]
        if dist.get_rank() == src:
            obj_list[0] = obj
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    # fallback: pickle to bytes, broadcast size then bytes into a tensor
    if dist.get_rank() == src:
        pick = pickle.dumps(obj)
        pick_tensor = torch.ByteTensor(list(pick)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        size_tensor = torch.LongTensor([pick_tensor.numel()]).to(pick_tensor.device)
    else:
        pick_tensor = None
        size_tensor = torch.LongTensor([0]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # broadcast the size
    dist.broadcast(size_tensor, src=src)
    sz = int(size_tensor.item())

    # create buffer
    if dist.get_rank() != src:
        pick_tensor = torch.empty(sz, dtype=torch.uint8, device=size_tensor.device)
    else:
        # if src has larger tensor, ensure pick_tensor length matches sz
        if pick_tensor.numel() != sz:
            pick_tensor = pick_tensor[:sz]

    dist.broadcast(pick_tensor, src=src)
    data = bytes(pick_tensor.tolist())
    return pickle.loads(data)
