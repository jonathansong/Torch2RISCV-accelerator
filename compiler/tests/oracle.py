"""Reference interpreter of IREE dispatch functions with the sa device's
numerics (docs/iree_compiler_plan.md §6.8 H): the semantic oracle of the sa
backend, independent of its code generator.

A dispatch source (iree-compile --iree-hal-dump-executable-sources-to) is
parsed with the MLIR Python bindings of the IREE compiler and executed on
byte buffers standing for the bindings:
- fp32 is the device's arithmetic (llm/fp32.py: IEEE round to nearest even,
  subnormals flushed), on uint32 bit patterns;
- math.exp, math.rsqrt and 1 / x (arith.divf with the constant 1.0 as the
  dividend) are the SFU functions (llm/sfu.py), as DeviceModel(SfuExact);
- fp32 sum reductions run in the device's order (word by word, then a lane
  tree; ref_model.DeviceModel.red_sum), max reductions are order-free;
- integer arithmetic is exact with the IR's widths.
Other operations are interpreted literally; an operation the oracle does
not know raises NotImplementedError (the backend should not accept it either).

    run_dispatch(func_op, buffers, constants, d)   buffers: {binding: bytearray}
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))
import fp32 as F  # noqa: E402
import sfu  # noqa: E402

from iree.compiler import ir  # noqa: E402

ONE = np.uint32(0x3F800000)


# --------------------------------------------------------------------------- values
class T:
    """A tensor or scalar value: arr (uint32 bits for f32, int64 for integers /
    index, bool for i1), and its element type name ('f32', 'i1', 'i8', ... 'index')."""

    def __init__(self, arr, et):
        self.arr, self.et = np.asarray(arr), et

    def __repr__(self):
        return f"T({self.et}, {self.arr.shape})"


def etype(t):
    s = str(t)
    return s


def elem_of(tensor_type):
    return str(ir.RankedTensorType(tensor_type).element_type)


def width(et):
    return {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64, "index": 64}[et]


def wrap(v, et):
    """Two's complement wrap of integer values to the type's width (signed view)."""
    w = width(et)
    if w == 1:
        return np.asarray(v).astype(bool)
    if w == 64:
        return np.asarray(v, np.int64)
    m = 1 << w
    v = np.asarray(v, np.int64) & (m - 1)
    return np.where(v >= m >> 1, v - m, v).astype(np.int64)


def f32_bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def f32_val(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


# --------------------------------------------------------------------------- buffers
NP = {"f32": np.float32, "i8": np.int8, "i32": np.int32, "i64": np.int64, "i16": np.int16, "i1": np.uint8}


def read_tensor(buf, offset, shape, et):
    dt = np.dtype(NP[et])
    n = int(np.prod(shape)) if shape else 1
    raw = np.frombuffer(bytes(buf[offset:offset + n * dt.itemsize]), dt).reshape(shape)
    if et == "f32":
        return T(raw.view(np.uint32).copy(), et)
    return T(raw.astype(np.int64), et)


def write_tensor(buf, offset, value, et):
    dt = np.dtype(NP[et])
    if et == "f32":
        data = np.asarray(value.arr, np.uint32).astype("<u4").tobytes()
    else:
        data = wrap(value.arr, et).astype(dt).tobytes()
    buf[offset:offset + len(data)] = data


# --------------------------------------------------------------------------- interpreter
class Interp:
    def __init__(self, buffers, constants, d):
        self.buffers, self.constants, self.d = buffers, constants, d
        self.env = {}
        self.subspans = {}           # value -> (binding, offset, shape template)

    def val(self, v):
        return self.env[v]

    def set(self, v, x):
        self.env[v] = x

    def run_block(self, block):
        for op in block.operations:
            self.run_op(op.operation)

    # ---------------------------------------------------------------- ops
    def run_op(self, op):
        name = op.name
        f = getattr(self, "op_" + name.replace(".", "_"), None)
        if f is None:
            raise NotImplementedError(f"oracle: {name}")
        f(op)

    def attr(self, op, name):
        return op.attributes[name]

    def op_arith_constant(self, op):
        a = op.attributes["value"]
        t = str(op.results[0].type)
        if isinstance(a, ir.FloatAttr):
            self.set(op.results[0], T(f32_bits(ir.FloatAttr(a).value), "f32"))
        elif isinstance(a, ir.IntegerAttr):
            self.set(op.results[0], T(np.int64(ir.IntegerAttr(a).value), t))
        elif isinstance(a, ir.DenseElementsAttr):
            rt = ir.RankedTensorType(op.results[0].type)
            et = str(rt.element_type)
            arr = np.array(ir.DenseElementsAttr(a)) if not ir.DenseElementsAttr(a).is_splat else \
                np.full(rt.shape, np.array(ir.DenseElementsAttr(a).get_splat_value().value))
            self.set(op.results[0], T(f32_bits(arr) if et == "f32" else np.asarray(arr, np.int64), et))
        else:
            raise NotImplementedError(f"oracle: constant {a}")

    def op_hal_interface_constant_load(self, op):
        ordinal = ir.IntegerAttr(op.attributes["ordinal"]).value
        self.set(op.results[0], T(np.int64(self.constants[ordinal]), "i32"))

    def op_util_assume_int(self, op):
        for r, o in zip(op.results, op.operands):
            self.set(r, self.val(o))

    def op_iree_tensor_ext_dispatch_workload_ordinal(self, op):
        self.set(op.results[0], self.val(op.operands[0]))

    def _cast_int(self, op):
        x = self.val(op.operands[0])
        self.set(op.results[0], T(wrap(x.arr, str(op.results[0].type)), str(op.results[0].type)))

    def op_arith_index_castui(self, op):
        x = self.val(op.operands[0])
        w = width(x.et)
        v = np.asarray(x.arr, np.int64) & ((1 << w) - 1) if w < 64 else x.arr
        self.set(op.results[0], T(v, str(op.results[0].type)))

    op_arith_index_cast = _cast_int
    op_arith_extsi = _cast_int
    op_arith_trunci = _cast_int

    def op_arith_extui(self, op):
        x = self.val(op.operands[0])
        w = width(x.et)
        self.set(op.results[0], T(np.asarray(x.arr, np.int64) & ((1 << w) - 1), str(op.results[0].type)))

    def _int_bin(fn):
        def f(self, op):
            a, b = self.val(op.operands[0]), self.val(op.operands[1])
            et = str(op.results[0].type)
            self.set(op.results[0], T(wrap(fn(np.asarray(a.arr, np.int64), np.asarray(b.arr, np.int64)), et), et))
        return f

    op_arith_addi = _int_bin(lambda a, b: a + b)
    op_arith_subi = _int_bin(lambda a, b: a - b)
    op_arith_muli = _int_bin(lambda a, b: a * b)
    op_arith_shli = _int_bin(lambda a, b: a << b)
    op_arith_ori = _int_bin(lambda a, b: a | b)
    op_arith_andi = _int_bin(lambda a, b: a & b)
    op_arith_divsi = _int_bin(lambda a, b: np.trunc(a / b).astype(np.int64))
    op_arith_remsi = _int_bin(lambda a, b: a - np.trunc(a / b).astype(np.int64) * b)

    def op_arith_cmpi(self, op):
        pred = str(op.attributes["predicate"])
        a, b = np.asarray(self.val(op.operands[0]).arr, np.int64), np.asarray(self.val(op.operands[1]).arr, np.int64)
        # predicate enum: eq 0, ne 1, slt 2, sle 3, sgt 4, sge 5, ult 6, ule 7, ugt 8, uge 9
        p = int(pred.split(":")[0]) if pred[0].isdigit() else pred
        fns = {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}
        if p not in fns:
            raise NotImplementedError(f"oracle: cmpi {pred}")
        self.set(op.results[0], T(fns[p], "i1"))

    def op_arith_cmpf(self, op):
        pred = str(op.attributes["predicate"])
        p = int(pred.split(":")[0])
        a, b = f32_val(self.val(op.operands[0]).arr), f32_val(self.val(op.operands[1]).arr)
        with np.errstate(invalid="ignore"):
            uno = np.isnan(a) | np.isnan(b)
            # 1 oeq 2 ogt 3 oge 4 olt 5 ole 6 one 7 ord 8 ueq 9 ugt 10 uge 11 ult 12 ule 13 une 14 uno
            r = {1: a == b, 2: a > b, 3: a >= b, 4: a < b, 5: a <= b, 6: (a != b) & ~uno, 7: ~uno,
                 8: (a == b) | uno, 9: (a > b) | uno, 10: (a >= b) | uno, 11: (a < b) | uno, 12: (a <= b) | uno,
                 13: (a != b) | uno, 14: uno}.get(p)
        if r is None:
            raise NotImplementedError(f"oracle: cmpf {pred}")
        self.set(op.results[0], T(r, "i1"))

    def op_arith_select(self, op):
        c, a, b = (self.val(o) for o in op.operands)
        self.set(op.results[0], T(np.where(np.asarray(c.arr, bool), a.arr, b.arr), a.et))

    def _f_bin(fn):
        def f(self, op):
            a, b = self.val(op.operands[0]), self.val(op.operands[1])
            self.set(op.results[0], T(fn(a.arr, b.arr), "f32"))
        return f

    op_arith_addf = _f_bin(F.add)
    op_arith_subf = _f_bin(F.sub)
    op_arith_mulf = _f_bin(F.mul)
    op_arith_maximumf = _f_bin(F.fmax)
    op_arith_minimumf = _f_bin(F.fmin)

    def op_arith_divf(self, op):
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        if not np.all(np.asarray(a.arr, np.uint32) == ONE):
            raise NotImplementedError("oracle: divf other than 1 / x (the device has only a reciprocal)")
        self.set(op.results[0], T(sfu.recip(np.asarray(b.arr, np.uint32)), "f32"))

    def op_arith_negf(self, op):
        self.set(op.results[0], T(F.neg(self.val(op.operands[0]).arr), "f32"))

    def op_math_absf(self, op):
        self.set(op.results[0], T(sfu.fabs(self.val(op.operands[0]).arr), "f32"))

    def op_math_exp(self, op):
        self.set(op.results[0], T(sfu.exp(np.asarray(self.val(op.operands[0]).arr, np.uint32)), "f32"))

    def op_math_rsqrt(self, op):
        self.set(op.results[0], T(sfu.rsqrt(np.asarray(self.val(op.operands[0]).arr, np.uint32)), "f32"))

    def op_math_roundeven(self, op):
        x = f32_val(self.val(op.operands[0]).arr)
        self.set(op.results[0], T(f32_bits(np.round(x)), "f32"))     # numpy rounds half to even

    def op_arith_sitofp(self, op):
        x = self.val(op.operands[0])
        self.set(op.results[0], T(F.from_int(np.asarray(x.arr, np.int64)), "f32"))

    def op_arith_fptosi(self, op):
        x = f32_val(self.val(op.operands[0]).arr).astype(np.float64)
        et = str(op.results[0].type)
        if np.any(np.isnan(x)) or np.any(np.abs(x) >= 2.0 ** (width(et) - 1)):
            raise ValueError("oracle: fptosi out of range (undefined)")
        self.set(op.results[0], T(np.trunc(x).astype(np.int64), et))

    # ---------------------------------------------------------------- HAL / tensors
    def op_hal_interface_binding_subspan(self, op):
        binding = ir.IntegerAttr(op.attributes["binding"]).value
        offset = 0
        if len(op.operands) and "byte_offset" not in ():
            # operands: [byte_offset?, dynamic_dims...]; the offset is the first
            # operand when the op has one (operandSegmentSizes)
            seg = op.attributes["operandSegmentSizes"] if "operandSegmentSizes" in op.attributes else None
            nofs = 1
            if seg is not None:
                nofs = list(ir.DenseI32ArrayAttr(seg))[0]
            if nofs:
                offset = int(self.val(op.operands[0]).arr)
            dyn = [int(self.val(o).arr) for o in list(op.operands)[nofs:]]
        else:
            dyn = []
        self.set(op.results[0], ("subspan", binding, offset, dyn))

    def _dispatch_type(self, v):
        t = str(v.type)                      # !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x?xf32>>
        inner = t[t.index("tensor<", t.index(":")) + 7:t.rindex(">") - 1]
        dims, et = inner.rsplit("x", 1) if "x" in inner else ("", inner)
        return [None if x == "?" else int(x) for x in dims.split("x")] if dims else [], et

    def _slice_args(self, op, first):
        """(offsets, sizes, strides) with the dynamic ones filled in."""
        vals = [int(np.asarray(self.val(o).arr)) for o in list(op.operands)[first:]]
        res = []
        for name in ("static_offsets", "static_sizes", "static_strides"):
            st = list(ir.DenseI64ArrayAttr(op.attributes[name]))
            out = []
            for s in st:
                if s == -9223372036854775808:          # ShapedType::kDynamic
                    out.append(vals.pop(0))
                else:
                    out.append(s)
            res.append(out)
        return res

    def _full(self, sub, v):
        _, binding, offset, dyn = sub
        dims, et = self._dispatch_type(v)
        dims = [dyn.pop(0) if x is None else x for x in dims] if dyn else dims
        return binding, offset, dims, et

    def op_iree_tensor_ext_dispatch_tensor_load(self, op):
        src = op.operands[0]
        sub = self.val(src)
        seg = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
        binding, offset, dims, et = self._full((sub[0], sub[1], sub[2], list(sub[3])), src)
        first = seg[0] + seg[1]
        offs, sizes, strides = self._slice_args(op, first)
        full = read_tensor(self.buffers[binding], offset, dims, et)
        sl = tuple(slice(o, o + s * st, st) for o, s, st in zip(offs, sizes, strides))
        arr = full.arr[sl] if dims else full.arr
        rt = ir.RankedTensorType(op.results[0].type)
        dims = list(np.asarray(arr).shape)
        while len(dims) > rt.rank:                    # rank-reducing load: drop unit dimensions
            dims.remove(1)
        self.set(op.results[0], T(np.asarray(arr).reshape(dims), et))

    def op_iree_tensor_ext_dispatch_tensor_store(self, op):
        value = self.val(op.operands[0])
        tgt = op.operands[1]
        sub = self.val(tgt)
        seg = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
        binding, offset, dims, et = self._full((sub[0], sub[1], sub[2], list(sub[3])), tgt)
        offs, sizes, strides = self._slice_args(op, seg[0] + seg[1] + seg[2])
        full = read_tensor(self.buffers[binding], offset, dims, et)
        sl = tuple(slice(o, o + s * st, st) for o, s, st in zip(offs, sizes, strides))
        if dims:
            full.arr[sl] = np.asarray(value.arr).reshape(full.arr[sl].shape)
        else:
            full.arr = np.asarray(value.arr)
        write_tensor(self.buffers[binding], offset, full, et)

    def op_tensor_empty(self, op):
        rt = ir.RankedTensorType(op.results[0].type)
        dyn = [int(np.asarray(self.val(o).arr)) for o in op.operands]
        shape = [dyn.pop(0) if s < 0 else s for s in rt.shape]
        et = str(rt.element_type)
        self.set(op.results[0], T(np.zeros(shape, np.uint32 if et == "f32" else np.int64), et))

    def op_linalg_fill(self, op):
        v, out = self.val(op.operands[0]), self.val(op.operands[1])
        self.set(op.results[0], T(np.full(out.arr.shape, v.arr, dtype=out.arr.dtype), out.et))

    def op_tensor_extract(self, op):
        t = self.val(op.operands[0])
        idx = [np.asarray(self.val(o).arr, np.int64) for o in list(op.operands)[1:]]
        self.set(op.results[0], T(t.arr[tuple(idx)], t.et))

    def op_linalg_index(self, op):
        dim = ir.IntegerAttr(op.attributes["dim"]).value
        self.set(op.results[0], T(self.index_vals[dim], "index"))

    def op_linalg_yield(self, op):
        self.yielded = [self.val(o) for o in op.operands]

    op_iree_linalg_ext_yield = op_linalg_yield

    def op_func_return(self, op):
        pass

    # ---------------------------------------------------------------- linalg.generic
    def op_linalg_generic(self, op):
        maps = [ir.AffineMapAttr(m).value for m in ir.ArrayAttr(op.attributes["indexing_maps"])]
        iters = [str(i) for i in ir.ArrayAttr(op.attributes["iterator_types"])]
        seg = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
        ins = [self.val(o) for o in list(op.operands)[:seg[0]]]
        outs = [self.val(o) for o in list(op.operands)[seg[0]:seg[0] + seg[1]]]
        nloops = maps[0].n_dims
        # loop ranges from the operand shapes
        ranges = [None] * nloops
        for m, t in zip(maps, ins + outs):
            for r, e in enumerate(m.results):
                s = str(e)
                if s.startswith("d") and s[1:].isdigit():
                    ranges[int(s[1:])] = t.arr.shape[r]
        if any(r is None for r in ranges):
            raise NotImplementedError("oracle: loop range not given by an operand dimension")
        red = [i for i, it in enumerate(iters) if "reduction" in it]
        grid = np.indices(ranges) if nloops else np.zeros((0,), np.int64)
        self.index_vals = [grid[i] for i in range(nloops)]

        def gather(t, m):
            if len(m.results) == 0:
                return np.broadcast_to(t.arr, ranges) if nloops else t.arr
            idx = []
            for e in m.results:
                s = str(e)
                if not (s.startswith("d") and s[1:].isdigit()):
                    raise NotImplementedError(f"oracle: indexing expression {s}")
                idx.append(self.index_vals[int(s[1:])])
            return t.arr[tuple(idx)]

        block = op.regions[0].blocks[0]
        args = list(block.arguments)
        for a, t, m in zip(args, ins + outs, maps):
            self.set(a, T(gather(t, m), t.et))
        if not red:
            self.run_block(block)
            outs_new = []
            for y, t, m in zip(self.yielded, outs, maps[len(ins):]):
                arr = np.broadcast_to(np.asarray(y.arr), ranges) if nloops else np.asarray(y.arr)
                res = [str(e) for e in m.results]
                if res == [f"d{i}" for i in range(nloops)]:
                    outs_new.append(T(arr.copy(), y.et))
                    continue
                if sorted(res) != [f"d{i}" for i in range(nloops)]:
                    raise NotImplementedError("oracle: output map is not a permutation")
                perm = [int(r[1:]) for r in res]              # out[d_perm0, d_perm1, ...] = value(d)
                outs_new.append(T(np.transpose(arr, perm).copy(), y.et))
            for r, t in zip(op.results, outs_new):
                self.set(r, t)
            return
        # reduction: out = combiner(elem, out); elem computed over the whole grid
        yops = list(block.operations)
        yieldop = yops[-1].operation
        if len(outs) != 1:
            raise NotImplementedError("oracle: multi-result reduction")
        comb = yieldop.operands[0].owner
        out_arg = args[-1]
        elem_v = [o for o in comb.operands if o != out_arg]
        if len(elem_v) != 1 or len(list(comb.operands)) != 2:
            raise NotImplementedError("oracle: reduction combiner does not read the accumulator once")
        # evaluate everything except the combiner (the accumulator arg is not used elsewhere)
        for o in yops[:-1]:
            if o.operation == comb:
                continue
            self.run_op(o.operation)
        elem = self.val(elem_v[0])
        earr = np.broadcast_to(np.asarray(elem.arr), ranges)
        out = outs[0]
        omap = maps[-1]
        keep = [int(str(e)[1:]) for e in omap.results]
        # move reduced dims last, in loop order, and flatten them
        perm = keep + red
        e2 = np.transpose(earr, perm).reshape([ranges[k] for k in keep] + [-1])
        name = comb.name
        if name == "arith.addf":
            acc = red_sum(e2, self.d)
            acc = F.add(acc, np.asarray(out.arr, np.uint32)) if not np.all(np.asarray(out.arr) == 0) else acc
        elif name == "arith.maximumf":
            acc = e2[..., 0]
            for i in range(1, e2.shape[-1]):
                acc = F.fmax(acc, e2[..., i])
            acc = F.fmax(acc, np.asarray(out.arr, np.uint32))
        elif name == "arith.addi":
            acc = wrap(np.sum(np.asarray(e2, np.int64), axis=-1) + np.asarray(out.arr, np.int64), out.et)
        else:
            raise NotImplementedError(f"oracle: reduction with {name}")
        self.set(op.results[0], T(np.asarray(acc).reshape(out.arr.shape), out.et))

    def op_linalg_batch_matmul(self, op):
        a, b, c = (self.val(o) for o in op.operands)
        acc = np.einsum("bij,bjk->bik", np.asarray(a.arr, np.int64), np.asarray(b.arr, np.int64))
        self.set(op.results[0], T(wrap(acc + np.asarray(c.arr, np.int64), c.et), c.et))

    def op_iree_linalg_ext_scatter(self, op):
        upd, idx, orig = (self.val(o) for o in op.operands)
        out = np.array(orig.arr, copy=True)
        for i, row in enumerate(np.asarray(idx.arr, np.int64).reshape(-1)):
            out[row] = upd.arr[i]
        self.set(op.results[0], T(out, orig.et))


def red_sum(e2, d):
    """fp32 sums over the last axis in the device's order (DeviceModel.red_sum):
    words of D lanes accumulated in order, then a lane tree; +0 padding."""
    n = e2.shape[-1]
    pad = (-n) % d
    g = np.concatenate([np.asarray(e2, np.uint32), np.zeros(e2.shape[:-1] + (pad,), np.uint32)], -1)
    g = g.reshape(e2.shape[:-1] + (-1, d))
    acc = g[..., 0, :]
    for i in range(1, g.shape[-2]):
        acc = F.add(acc, g[..., i, :])
    while acc.shape[-1] > 1:
        acc = F.add(acc[..., 0::2], acc[..., 1::2])
    return acc[..., 0]


def load_dispatch(path):
    """(context, module, func op) of a dumped dispatch source."""
    ctx = ir.Context()
    m = ir.Module.parse(open(path).read(), ctx)
    funcs = []

    def walk(op):
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    if o.operation.name == "func.func":
                        funcs.append(o.operation)
                    else:
                        walk(o.operation)
    walk(m.operation)
    if len(funcs) != 1:
        raise ValueError(f"{path}: {len(funcs)} functions")
    return ctx, m, funcs[0]


def run_dispatch(func, buffers, constants, d):
    """Executes the dispatch function on the binding buffers (in place)."""
    it = Interp(buffers, constants, d)
    it.run_block(func.regions[0].blocks[0])
