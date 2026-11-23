"""Private eager IPC implementations of the unscaled fused TP operators."""

import math
import os

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed import _symmetric_memory as sm


def requested(x: torch.Tensor) -> bool:
    # Never select or change an application's symmetric allocator here.
    return (
        torch.version.hip is not None
        and x.is_cuda
        and os.environ.get("TORCH_SYMM_MEM_ROCSHMEM_ASYNC_TP") == "1"
        and sm.get_backend(x.device) == "NVSHMEM"
    )


def pieces_for(
    x: torch.Tensor, dim: int, group_name: c10d.GroupName, *, scatter_cols: int | None = None
) -> int:
    group = c10d._resolve_process_group(group_name)
    if (
        group is not c10d.group.WORLD
        or group.size() < 2
        or not 0 <= dim < x.ndim - 1
        or x.dtype not in (torch.float32, torch.bfloat16)
        or x.numel() == 0
        or torch.cuda.is_current_stream_capturing()
    ):
        return 0
    # Keep tail handling and the existing small-M native path out of this
    # experimental selector. Only collective-uniform shape metadata is used.
    rows = math.prod(x.shape[:-1])
    if scatter_cols is not None:
        if x.size(dim) % group.size():
            return 0
        rows //= group.size()
    if rows * group.size() <= 4096 or rows % 4:
        return 0
    # Large local shards can expose their first GEMM sooner without making the
    # individual SDMA puts small. RS benefits from larger producer GEMMs.
    if group.size() == 8:
        if scatter_cols is None:
            if rows % 8 == 0 and x.nbytes >= (64 << 20):
                return 8
        else:
            # Size RS tiles by the collective-uniform output, not the local
            # contracting dimension. Keep about 16 MiB per peer per transfer.
            size = rows * scatter_cols * x.element_size()
            target = min(8, (size + (8 << 20)) // (16 << 20))
            for count in range(target, 4, -1):
                if rows % count == 0:
                    return count
    return 4


def gather_mm(
    shard: torch.Tensor,
    bs: list[torch.Tensor],
    dim: int,
    group_name: c10d.GroupName,
    pieces: int,
    return_a: bool,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    x = shard.movedim(dim, 0)
    flat = x.contiguous().view(-1, x.size(-1))
    rows, k = flat.shape
    with torch.cuda.device(x.device):
        group = c10d._resolve_process_group(group_name)
        p = group.size()
        hdl = sm.get_symm_mem_workspace(group_name, p * flat.nbytes)
        gathered = hdl.get_buffer(group.rank(), (p, rows, k), x.dtype)
        ready = torch.zeros(pieces, dtype=torch.int64, device=x.device)
        compute = torch.cuda.current_stream()
        comm = sm._get_backend_stream()
        comm.wait_stream(compute)
        with torch.cuda.stream(comm):
            torch.ops.symm_mem._rocshmem_gather_slices(flat, gathered, ready, group_name)
        dests = [x.new_empty(p, rows, b.size(1)) for b in bs]
        m = rows // pieces
        for i in range(pieces):
            torch.ops.symm_mem.nvshmem_wait_for_signal(ready[i : i + 1], 1, group.rank())
            region = slice(i * m, (i + 1) * m)
            for b, dest in zip(bs, dests):
                torch.bmm(gathered[:, region], b.expand(p, -1, -1), out=dest[:, region])
        compute.wait_stream(comm)

        def restore(t: torch.Tensor) -> torch.Tensor:
            return t.view(p * x.size(0), *x.shape[1:-1], t.size(-1)).movedim(0, dim)

        # A caller retaining A must not observe writes by the next collective.
        a = restore(gathered.clone()) if return_a else None
        comm.wait_stream(compute)
        return a, [restore(dest) for dest in dests]


def mm_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int,
    reduction: str,
    group_name: c10d.GroupName,
    pieces: int,
) -> torch.Tensor:
    x = a.movedim(dim, 0).contiguous()
    p = c10d._get_group_size_by_name(group_name)
    tiles = x.view(p, -1, x.size(-1))
    rows, n = tiles.size(1), b.size(1)
    m = rows // pieces
    with torch.cuda.device(x.device):
        compute = torch.cuda.current_stream()
        comm = sm._get_backend_stream()
        # One send and receive region per piece. The next GEMM may start
        # before this piece has been sent; it must not overwrite its source.
        elements = p * m * n
        hdl = sm.get_symm_mem_workspace(
            group_name, 2 * pieces * elements * x.element_size()
        )
        result = a.new_empty(rows, n)
        def reduce_into(partial: torch.Tensor, dest: torch.Tensor) -> None:
            if sm.triton_reduce_partials_first_dim(
                partial, reduce_op=reduction, output_dtype=a.dtype, group_size=p, out=dest
            ) is None:
                op = torch.sum if reduction == "sum" else torch.mean
                op(partial, dim=0, out=dest)
        # The inputs and outputs of different pieces are disjoint. Rendezvous
        # once before overwriting this invocation's workspace, not per piece.
        compute.wait_stream(comm)
        hdl.barrier()
        pending = None
        for i in range(pieces):
            send = hdl.get_buffer(hdl.rank, (p, m, n), a.dtype, 2 * i * elements)
            recv = hdl.get_buffer(hdl.rank, (p, m, n), a.dtype, (2 * i + 1) * elements)
            torch.bmm(tiles[:, i * m : (i + 1) * m], b.expand(p, -1, -1), out=send)
            comm.wait_stream(compute)
            with torch.cuda.stream(comm):
                torch.ops.symm_mem.nvshmem_all_to_all(send, recv, group_name)
                done = comm.record_event()
            if pending is not None:
                previous, ready = pending
                compute.wait_event(ready)
                reduce_into(previous, result[(i - 1) * m : i * m])
            pending = recv, done
        assert pending is not None
        previous, ready = pending
        compute.wait_event(ready)
        reduce_into(previous, result[-m:])
        comm.wait_stream(compute)
        return result.view(x.size(0) // p, *x.shape[1:-1], n).movedim(0, dim).contiguous()
