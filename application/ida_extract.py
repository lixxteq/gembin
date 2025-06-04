# encoding:utf-8
# extract features from executable binary files using IDA Pro and Python 3

import sys
import os
import time
import json
import subprocess

from idautils import *
from idaapi import *
from idc import *
import networkx as nx
from LogRecorder import CLogRecoder

# --- Instruction categories for different architectures ---
TRANSFER_X86 = ['MOV', 'PUSH', 'POP', 'XCHG', 'IN', 'OUT', 'XLAT', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF', 'PUSHF', 'POPF']
ARITH_X86 = ['ADD', "ADC", "ADCX", "ADOX", "SBB", 'SUB', 'MUL', 'DIV', 'INC', 'DEC', 'IMUL', 'IDIV', 'CMP', "NEG",
             "DAA", "DAS", "AAA", "AAS", "AAM", "AAD"]
TRANSFER_ARM = {"B", "BAL", "BNE", "BEQ", "BPL", "BMI", "BCC", "BLO", "BCS", "BHS", "BVC", "BVS", "BGT", "BGE", "BLT", "BLE", "BHI", "BLS"}
ARITH_ARM = {"add", "adc", "qadd", "sub", "sbc", "rsb", "qsub", "mul", "mla", "mls", "umull", "umlal", "smull",
             "smlal", "udiv", "sdiv", "cmp", "cmn", "tst"}
TRANSFER_MIPS = {"beqz", "beq", "bne", "bgez", "b", "bnez", "bgtz", "bltz", "blez", "bgt", "bge", "blt", "ble", "bgtu", "bgeu", "bltu", "bleu"}
ARITH_MIPS = {"add", "addu", "addi", "addiu", "and", "andi", "div", "divu", "mult", "multu", "slt", "sltu", "slti", "sltiu"}
TRANSFER_PPC = {"b", "blt", "beq", "bge", "bgt", "blr", "bne"}
ARITH_PPC = {"add", "addi", "addme", "addze", "neg", "subf", "subfic", "subfme", "subze", "mulhw", "mulli",
             "mullw", "divw", "cmp", "cmpi", "cmpl", "cmpli"}

# General instruction categories (used in feature extraction)
TRANSFER_INS = ['MOV', 'PUSH', 'POP', 'XCHG', 'IN', 'OUT', 'XLAT', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF', 'PUSHF', 'POPF']
ARITH_INS = ['ADD', 'SUB', 'MUL', 'DIV', 'XOR', 'INC', 'DEC', 'IMUL', 'IDIV', 'OR', 'NOT', 'SLL', 'SRL']

# --- Constants and logger setup ---
OPTYPEOFFSET = 1000
IMM_MASK = 0xffffffff  # Immediate value mask

# Logger initialization
log_date = time.strftime("%Y-%m-%d", time.localtime())
logger = CLogRecoder(logfile=f'{log_date}.log')
logger.addStreamHandler()
logger.INFO("\n---------------------\n")
logger.INFO(f"IDA Version {IDA_SDK_VERSION}")

# --- IDA version compatibility ---
IDA700 = False
if IDA_SDK_VERSION >= 700:
    logger.INFO("Using IDA7xx API")
    IDA700 = True
    GetOpType = get_operand_type
    GetOperandValue = get_operand_value
    SegName = get_segm_name
    autoWait = auto_wait
    GetFunctionName = get_func_name
    import ida_pro
    Exit = ida_pro.qexit
else:
    # For older IDA versions, use legacy names
    GetOpType = GetOpType
    GetOperandValue = GetOperandValue
    SegName = SegName
    autoWait = autoWait
    GetFunctionName = GetFunctionName
    Exit = Exit

def wait_for_ida_analysis():
    autoWait()

wait_for_ida_analysis()

def get_elf_bits(filename):
    """detect ELF bitness (32/64) using the 'file' command."""
    cmd = f'file -b {filename}'
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        bits = output.split(' ')[1]
        return 32 if int(bits[:1]) == 32 else 64
    except subprocess.CalledProcessError as e:
        print('error', e.returncode, e.output)
        return 64  # Default to 64-bit if detection fails

# Adjust immediate mask for 64-bit binaries
if get_elf_bits(get_input_file_path()) == 64:
    IMM_MASK = 0xffffffffffffffff

# --- Feature extraction helpers ---

def extract_string_constants(start_ea, end_ea):
    """extract string constants from a basic block."""
    strings = []
    for head in Heads(start_ea, end_ea):
        for i in range(2):
            if GetOpType(head, i) == o_imm:
                imm_value = GetOperandValue(head, i) & IMM_MASK
                if is_strlit(get_flags(imm_value)):
                    string_val = get_strlit_contents(imm_value, -1, STRTYPE_C)
                    if string_val is not None:
                        strings.append(string_val.decode('utf-8', errors='ignore'))
    return strings

def extract_numeric_constants(start_ea, end_ea):
    """extract numeric (non-string) constants from a basic block."""
    constants = []
    for head in Heads(start_ea, end_ea):
        for i in range(2):
            if GetOpType(head, i) == o_imm:
                imm_value = GetOperandValue(head, i) & IMM_MASK
                if not is_strlit(get_flags(imm_value)):
                    constants.append(imm_value)
    return constants

def extract_call_instructions(start_ea, end_ea):
    """extract call instructions from a basic block."""
    return [head for head in Heads(start_ea, end_ea) if is_call_insn(head)]

def extract_transfer_instructions(start_ea, end_ea):
    """extract transfer instructions from a basic block."""
    return [head for head in Heads(start_ea, end_ea) if print_insn_mnem(head) in TRANSFER_INS]

def extract_arithmetic_instructions(start_ea, end_ea):
    """extract arithmetic instructions from a basic block."""
    return [head for head in Heads(start_ea, end_ea) if print_insn_mnem(head) in ARITH_INS]

def count_instructions(start_ea, end_ea):
    """count the number of instructions in a basic block."""
    return sum(1 for head in Heads(start_ea, end_ea) if is_code(get_flags(head)))

def count_offspring(start_ea, end_ea):
    """count the number of flow offspring in a basic block."""
    return sum(1 for head in Heads(start_ea, end_ea) if is_flow(get_flags(head)))

def get_basic_blocks(func_addr):
    """get all basic blocks of a function."""
    func = get_func(func_addr)
    if func is None:
        return []
    return [(block.start_ea, block.end_ea) for block in FlowChart(func)]

def extract_function_features(func_addr, func_name=''):
    """extract features for a function and its basic blocks."""
    features = {}
    if not func_name:
        func_name = GetFunctionName(func_addr)
        logger.INFO(f"Processing function features: [{func_addr} {func_name}]")
    features['fun_name'] = func_name

    blocks = get_basic_blocks(func_addr)
    logger.INFO(f"Extracted basic blocks: {blocks}")

    # Build control flow graph (CFG)
    cfg = nx.DiGraph()
    for start_ea, end_ea in blocks:
        cfg.add_node(hex(start_ea))
        features[hex(start_ea)] = {
            "String_Constant": extract_string_constants(start_ea, end_ea),
            "Numberic_Constant": extract_numeric_constants(start_ea, end_ea),
            "No_Call": len(extract_call_instructions(start_ea, end_ea)),
            "No_Tran": len(extract_transfer_instructions(start_ea, end_ea)),
            "No_Arith": len(extract_arithmetic_instructions(start_ea, end_ea)),
            "No_Instru": count_instructions(start_ea, end_ea),
            "No_offspring": count_offspring(start_ea, end_ea),
            "pre": [],
            "suc": []
        }

    flow = ida_gdl.FlowChart(get_func(func_addr))
    for start_ea, end_ea in blocks:
        block = next((b for b in flow if b.start_ea == start_ea and b.end_ea == end_ea), None)
        if block:
            for succ in block.succs():
                cfg.add_edge(hex(start_ea), hex(succ.start_ea))
                features[hex(start_ea)]["suc"].append(hex(succ.start_ea))
                features[hex(succ.start_ea)]["pre"].append(hex(start_ea))
    return features

def get_all_function_addresses():
    """return addresses of all functions in the binary."""
    return list(Functions())

def find_function_by_name(func_name):
    for func in Functions():
        if GetFunctionName(func) == func_name:
            return func
    return None

def main():
    if len(idc.ARGV) < 2:
        print(f"Usage: {idc.ARGV[0]} <output_file> [function_name]")
        Exit(1)

    output_file = idc.ARGV[1]
    func_name = idc.ARGV[2] if len(idc.ARGV) > 2 else ''

    with open(output_file, 'w', encoding='utf-8') as f:
        if func_name:
            func_addr = find_function_by_name(func_name)
            if func_addr is None:
                print(f"Function {func_name} not found")
                Exit(1)
            features = extract_function_features(func_addr, func_name)
            f.write(json.dumps(features, ensure_ascii=False) + '\n')
        else:
            funcs = get_all_function_addresses()
            logger.INFO(f"Functions mappings: {funcs}")
            for func_addr in funcs:
                features = extract_function_features(func_addr)
                logger.INFO(f"Extracted features (block attributes): {features}")
                f.write(json.dumps(features, ensure_ascii=False) + '\n')

    print(f"Features written to {output_file}")
    Exit(0)

if __name__ == '__main__':
    main()