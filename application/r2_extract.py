#!/usr/bin/env python3
# encoding:utf-8
# extract features from executable binary files with r2pipe and radare2 driver
import sys

import r2pipe
# todo: dynamic path
sys.path.insert(0, r"/media/storage/Documents/reverse/vkr/diploma/.linux_venv/lib/python3.10/site-packages")
import os
import time
import json
import subprocess
import networkx as nx
from typing import List, Dict, Tuple, Optional

# instruction categories for different architectures
TRANSFER_X86 = ['mov', 'push', 'pop', 'xchg', 'in', 'out', 'xlat', 'lea', 'lds', 'les', 'lahf', 'sahf', 'pushf', 'popf']
ARITH_X86 = ['add', "adc", "adcx", "adox", "sbb", 'sub', 'mul', 'div', 'inc', 'dec', 'imul', 'idiv', 'cmp', "neg",
             "daa", "das", "aaa", "aas", "aam", "aad"]
TRANSFER_ARM = {"b", "bal", "bne", "beq", "bpl", "bmi", "bcc", "blo", "bcs", "bhs", "bvc", "bvs", "bgt", "bge", "blt", "ble", "bhi", "bls"}
ARITH_ARM = {"add", "adc", "qadd", "sub", "sbc", "rsb", "qsub", "mul", "mla", "mls", "umull", "umlal", "smull",
             "smlal", "udiv", "sdiv", "cmp", "cmn", "tst"}
TRANSFER_MIPS = {"beqz", "beq", "bne", "bgez", "b", "bnez", "bgtz", "bltz", "blez", "bgt", "bge", "blt", "ble", "bgtu", "bgeu", "bltu", "bleu"}
ARITH_MIPS = {"add", "addu", "addi", "addiu", "and", "andi", "div", "divu", "mult", "multu", "slt", "sltu", "slti", "sltiu"}
TRANSFER_PPC = {"b", "blt", "beq", "bge", "bgt", "blr", "bne"}
ARITH_PPC = {"add", "addi", "addme", "addze", "neg", "subf", "subfic", "subfme", "subze", "mulhw", "mulli",
             "mullw", "divw", "cmp", "cmpi", "cmpl", "cmpli"}

# general instruction categories (used in feature extraction)
TRANSFER_INS = ['mov', 'push', 'pop', 'xchg', 'in', 'out', 'xlat', 'lea', 'lds', 'les', 'lahf', 'sahf', 'pushf', 'popf']
ARITH_INS = ['add', 'sub', 'mul', 'div', 'xor', 'inc', 'dec', 'imul', 'idiv', 'or', 'not', 'sll', 'srl']

IMM_MASK = 0xffffffff  # immediate value mask

def get_elf_bits(filename: str) -> int:
    """Detect ELF bitness (32/64) using the 'file' command."""
    cmd = f'file -b "{filename}"'
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        if '64-bit' in output:
            return 64
        elif '32-bit' in output:
            return 32
        else:
            return 64  # default to 64-bit if unclear
    except subprocess.CalledProcessError as e:
        print(f'error detecting bitness: {e}')
        return 64  # default to 64-bit if detection fails

def extract_string_constants(r2, start_addr: int, end_addr: int) -> List[str]:
    """Extract string constants from a basic block."""
    strings = []

    # get disassembly for the range
    disasm = r2.cmdj(f"pdj @ {start_addr}~{end_addr - start_addr}")
    if not disasm:
        return strings
    
    for instr in disasm:
        if 'opcode' in instr:
            # check operands for string references
            if 'operands' in instr:
                for operand in instr['operands']:
                    if operand.get('type') == 'imm':
                        addr = operand.get('value', 0)
                        # check if this address contains a string
                        string_info = r2.cmdj(f"Csj @ {addr}")
                        if string_info and len(string_info) > 0:
                            for s in string_info:
                                if s.get('type') == 'string':
                                    strings.append(s.get('string', ''))
    
    return strings

def extract_numeric_constants(r2, start_addr: int, end_addr: int) -> List[int]:
    """Extract numeric (non-string) constants from a basic block."""
    constants = []
    
    # get disassembly for the range
    disasm = r2.cmdj(f"pdj @ {start_addr}~{end_addr - start_addr}")
    if not disasm:
        return constants
    
    for instr in disasm:
        if 'opcode' in instr and 'operands' in instr:
            for operand in instr['operands']:
                if operand.get('type') == 'imm':
                    value = operand.get('value', 0)
                    # check if it's not a string reference
                    string_info = r2.cmdj(f"Csj @ {value}")
                    if not string_info or len(string_info) == 0:
                        constants.append(value & IMM_MASK)
    
    return constants

def extract_call_instructions(r2, start_addr: int, end_addr: int) -> List[int]:
    """Extract call instructions from a basic block."""
    calls = []
    
    # get disassembly for the range
    disasm = r2.cmdj(f"pdj @ {start_addr}~{end_addr - start_addr}")
    if not disasm:
        return calls
    
    for instr in disasm:
        if 'type' in instr and instr['type'] in ['call', 'ucall']:
            calls.append(instr.get('offset', start_addr))
    
    return calls

def extract_transfer_instructions(r2, start_addr: int, end_addr: int) -> List[int]:
    """Extract transfer instructions from a basic block."""
    transfers = []
    
    # get disassembly for the range
    disasm = r2.cmdj(f"pdj @ {start_addr}~{end_addr - start_addr}")
    if not disasm:
        return transfers
    
    for instr in disasm:
        if 'opcode' in instr:
            mnemonic = instr['opcode'].split()[0].lower()
            if mnemonic in TRANSFER_INS:
                transfers.append(instr.get('offset', start_addr))
    
    return transfers

def extract_arithmetic_instructions(r2, start_addr: int, end_addr: int) -> List[int]:
    """Extract arithmetic instructions from a basic block."""
    arithmetic = []
    
    # get disassembly for the range
    disasm = r2.cmdj(f"pdj @ {start_addr}~{end_addr - start_addr}")
    if not disasm:
        return arithmetic
    
    for instr in disasm:
        if 'opcode' in instr:
            mnemonic = instr['opcode'].split()[0].lower()
            if mnemonic in ARITH_INS:
                arithmetic.append(instr.get('offset', start_addr))
    
    return arithmetic

def count_instructions(r2, start_addr: int, end_addr: int) -> int:
    """Count the number of instructions in a basic block."""
    # get disassembly for the range
    disasm = r2.cmdj(f"pdj @ {start_addr}~{end_addr - start_addr}")
    return len(disasm) if disasm else 0

def count_offspring(r2, start_addr: int, end_addr: int) -> int:
    """Count the number of flow offspring in a basic block."""
    bb_info = r2.cmdj(f"afbj @ {start_addr}")
    if bb_info and len(bb_info) > 0:
        for bb in bb_info:
            if bb.get('addr') == start_addr:
                jump_count = 1 if bb.get('jump') is not None else 0
                fail_count = 1 if bb.get('fail') is not None else 0
                return jump_count + fail_count
    return 0

def get_basic_blocks(r2, func_addr: int) -> List[Tuple[int, int]]:
    """Get all basic blocks of a function."""
    blocks = []
    
    # r2 function analysis
    func_info = r2.cmdj(f"afij @ {func_addr}")
    if not func_info or len(func_info) == 0:
        return blocks
    
    # basic blocks for this function
    bb_info = r2.cmdj(f"afbj @ {func_addr}")
    if bb_info:
        for bb in bb_info:
            start_addr = bb.get('addr', 0)
            size = bb.get('size', 0)
            end_addr = start_addr + size
            blocks.append((start_addr, end_addr))
    
    return blocks

def extract_function_features(r2, func_addr: int, func_name: str = '') -> Dict:
    """Extract features for a function and its basic blocks."""
    features = {}
    
    if not func_name:
        # function name from radare2
        func_info = r2.cmdj(f"afij @ {func_addr}")
        if func_info and len(func_info) > 0:
            func_name = func_info[0].get('name', f'sub_{func_addr:x}')
        else:
            func_name = f'sub_{func_addr:x}'
    
    # remove "sym." prefix if present (r2 specific)
    if func_name.startswith("sym."):
        func_name = func_name[4:]
    
    print(f"Processing function features: [{func_addr:x} {func_name}]")
    features['fun_name'] = func_name

    blocks = get_basic_blocks(r2, func_addr)
    print(f"Extracted basic blocks: {blocks}")

    # build control flow graph with nx
    cfg = nx.DiGraph()
    for start_addr, end_addr in blocks:
        cfg.add_node(hex(start_addr))
        features[hex(start_addr)] = {
            "String_Constant": extract_string_constants(r2, start_addr, end_addr),
            "Numberic_Constant": extract_numeric_constants(r2, start_addr, end_addr),
            "No_Call": len(extract_call_instructions(r2, start_addr, end_addr)),
            "No_Tran": len(extract_transfer_instructions(r2, start_addr, end_addr)),
            "No_Arith": len(extract_arithmetic_instructions(r2, start_addr, end_addr)),
            "No_Instru": count_instructions(r2, start_addr, end_addr),
            "No_offspring": count_offspring(r2, start_addr, end_addr),
            "pre": [],
            "suc": []
        }

    # add edges and update predecessor/successor info
    bb_info = r2.cmdj(f"afbj @ {func_addr}")
    if bb_info:
        for bb in bb_info:
            start_addr = bb.get('addr', 0)
            
            # add successors
            if 'jump' in bb and bb['jump']:
                succ_addr = bb['jump']
                cfg.add_edge(hex(start_addr), hex(succ_addr))
                if hex(start_addr) in features:
                    features[hex(start_addr)]["suc"].append(hex(succ_addr))
                if hex(succ_addr) in features:
                    features[hex(succ_addr)]["pre"].append(hex(start_addr))
            
            if 'fail' in bb and bb['fail']:
                succ_addr = bb['fail']
                cfg.add_edge(hex(start_addr), hex(succ_addr))
                if hex(start_addr) in features:
                    features[hex(start_addr)]["suc"].append(hex(succ_addr))
                if hex(succ_addr) in features:
                    features[hex(succ_addr)]["pre"].append(hex(start_addr))
    
    return features

def get_all_function_addresses(r2) -> List[int]:
    """Return addresses of all functions in the binary."""
    functions = []
    func_list = r2.cmdj("aflj")
    if func_list:
        for func in func_list:
            functions.append(func.get('offset', 0))
    return functions

def find_function_by_name(r2, func_name: str) -> Optional[int]:
    """Find function address by its name."""
    func_list = r2.cmdj("aflj")
    if func_list:
        for func in func_list:
            if func.get('name', '') == func_name:
                return func.get('offset', 0)
    return None

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <binary_file> <output_file> [function_name]")
        sys.exit(1)

    binary_file = sys.argv[1]
    output_file = sys.argv[2]
    func_name = sys.argv[3] if len(sys.argv) > 3 else ''

    # open binary in radare2
    r2 = r2pipe.open(binary_file)
    r2.cmd("aaa")  # analyze all (possibly do experimental aaaa?)
    
    # adjust immediate mask for 64-bit binaries
    global IMM_MASK
    if get_elf_bits(binary_file) == 64:
        IMM_MASK = 0xffffffffffffffff

    with open(output_file, 'w', encoding='utf-8') as f:
        if func_name:
            func_addr = find_function_by_name(r2, func_name)
            if func_addr is None:
                print(f"Function {func_name} not found")
                sys.exit(1)
            features = extract_function_features(r2, func_addr, func_name)
            f.write(json.dumps(features, ensure_ascii=False) + '\n')
        else:
            funcs = get_all_function_addresses(r2)
            print(f"Functions found: {len(funcs)}")
            for func_addr in funcs:
                features = extract_function_features(r2, func_addr)
                print(f"Extracted features for function at {func_addr:x}")
                f.write(json.dumps(features, ensure_ascii=False) + '\n')

    print(f"Features written to {output_file}")

if __name__ == '__main__':
    main()