"""Static SASS instruction counts of the compiled kernels.

Hardware performance counters are not accessible on the measurement host
(ERR_NVGPUCTRPERM).  For fully unrolled kernels the per-thread instruction
stream of an interior thread is fixed at compile time, so counting the
floating-point, memory and atomic instructions in the SASS the driver
actually executes gives an exact per-thread work measure without counters.
The kernels are compiled with the same NVRTC and options as the timed
kernels (cupy.cuda.compiler.compile_using_nvrtc, the path RawModule uses),
for the device's architecture, and disassembled with cuobjdump.

Classes counted (opcode prefix before the first '.'):
  fp64       DFMA DMUL DADD
  fp32       FFMA FMUL FADD (and their 32I immediate forms)
  ldg/stg    global loads/stores        lds/sts  shared loads/stores
  atom       global atomics/reductions (ATOM, ATOMG, RED)
  int        integer/address arithmetic (IMAD IADD3 LEA SHF LOP3 ISETP ...)
Backward branches are reported; FP instructions inside a loop would make the
static count a lower bound for the dynamic count.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter

from cupy.cuda import compiler

FP64 = {"DFMA", "DMUL", "DADD"}
FP32 = {"FFMA", "FMUL", "FADD", "FFMA32I", "FMUL32I", "FADD32I"}
INT = {"IMAD", "IADD3", "LEA", "SHF", "LOP3", "ISETP", "IABS", "IMNMX", "SEL", "IMUL", "IADD",
       "LEA.HI", "PRMT", "I2F", "F2I", "FLO", "POPC", "BMSK"}

_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)([.\w]*)\s*(.*?);")


def _cuobjdump():
    for base in [os.environ.get("CUDA_PATH", ""),
                 r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"]:
        p = os.path.join(base, "bin", "cuobjdump.exe")
        if base and os.path.isfile(p):
            return p
    p = shutil.which("cuobjdump")
    if not p:
        raise FileNotFoundError("cuobjdump not found")
    return p


def _tool(name):
    return os.path.join(os.path.dirname(_cuobjdump()), name)


def compile_sass(src: str, options=("-std=c++14",), arch="89"):
    """NVRTC (the compiler CuPy uses) -> PTX, then ptxas of the same CUDA
    release -> cubin -> cuobjdump -sass.  (cuobjdump cannot read the cubin
    NVRTC emits directly; the PTX route gives the identical backend.)
    Returns (sass_text, ptxas_info)."""
    from cupy_backends.cuda.libs import nvrtc
    prog = nvrtc.createProgram(src, "k.cu", [], [])
    try:
        nvrtc.compileProgram(prog, list(options) + [f"-arch=compute_{arch}"])
        ptx = nvrtc.getPTX(prog)
    finally:
        nvrtc.destroyProgram(prog)
    with tempfile.TemporaryDirectory() as d:
        pp, cb = os.path.join(d, "k.ptx"), os.path.join(d, "k.cubin")
        with open(pp, "wb") as f:
            f.write(ptx if isinstance(ptx, bytes) else ptx.encode())
        info = subprocess.run([_tool("ptxas.exe"), f"-arch=sm_{arch}", "-O3", "-v", pp, "-o", cb],
                              capture_output=True, text=True, timeout=300)
        if info.returncode != 0:
            raise RuntimeError(info.stderr)
        txt = subprocess.check_output([_cuobjdump(), "-sass", cb], text=True, timeout=120)
    return txt, (info.stdout or "") + (info.stderr or "")


def _ptxas_resources(info: str):
    regs = re.search(r"Used (\d+) registers", info)
    spill_st = re.search(r"(\d+) bytes spill stores", info)
    spill_ld = re.search(r"(\d+) bytes spill loads", info)
    smem = re.search(r"(\d+) bytes smem", info)
    return {"registers": int(regs.group(1)) if regs else None,
            "spill_store_bytes": int(spill_st.group(1)) if spill_st else 0,
            "spill_load_bytes": int(spill_ld.group(1)) if spill_ld else 0,
            "static_smem_bytes": int(smem.group(1)) if smem else 0}


def count(src: str, function: str | None = None):
    """Instruction class counts for one kernel (or the only one in src)."""
    txt, info = compile_sass(src)
    funcs, cur = {}, None
    for line in txt.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        mm = _LINE.search(line)
        if mm:
            funcs[cur].append((int(mm.group(1), 16), mm.group(3), mm.group(4), mm.group(5)))
    if function is None:
        if len(funcs) != 1:
            raise ValueError(f"several functions: {list(funcs)}")
        function = next(iter(funcs))
    ins = funcs[function]
    ops = Counter(op for _, op, _, _ in ins)
    backward = 0
    for addr, op, mod, rest in ins:
        if op == "BRA":
            t = re.search(r"0x([0-9a-f]+)", rest)
            if t and int(t.group(1), 16) <= addr:
                backward += 1
    c = {
        "total": len(ins),
        "fp64": sum(ops[o] for o in FP64),
        "fp32": sum(ops[o] for o in FP32),
        "dfma": ops["DFMA"], "dmul": ops["DMUL"], "dadd": ops["DADD"],
        "ffma": ops["FFMA"] + ops["FFMA32I"], "fmul": ops["FMUL"] + ops["FMUL32I"],
        "fadd": ops["FADD"] + ops["FADD32I"],
        "ldg": ops["LDG"], "stg": ops["STG"], "lds": ops["LDS"], "sts": ops["STS"],
        "ldc": ops["LDC"], "atom": ops["ATOM"] + ops["ATOMG"] + ops["RED"],
        "int": sum(v for k, v in ops.items() if k in INT),
        "backward_branches": backward,
        "function": function,
    }
    c["flop64"] = 2 * c["dfma"] + c["dmul"] + c["dadd"]
    c["flop32"] = 2 * c["ffma"] + c["fmul"] + c["fadd"]
    c.update(_ptxas_resources(info))
    return c
