# extracts function features from binaries or static archives using radare2 driver by default
import os
import sys
import time
import json
import argparse
import subprocess
import tempfile

# path to ida pro executable (only for ida driver)
# IDAPATH = r'D:\Documents\reverse\IDA-Pro-7.7\idat64.exe'
PRO_PATH = sys.path[0]
OVERWRITE = True

class FeatureExtractor:
    def __init__(self, binary_path):
        self.binary_path = binary_path
        self.tmpfile = os.path.join(PRO_PATH, os.path.basename(binary_path) + "tmp.json")

    def _read_features(self):
        """Yield features from the temporary JSON file."""
        with open(self.tmpfile, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line}")
                    continue

    def _delete_tmpfile(self):
        if os.path.exists(self.tmpfile):
            os.remove(self.tmpfile)

    def extract_all(self):
        """Extract features for all functions."""
        return self.extract_function('')

    def extract_function(self, func_name):
        """Extract features for a specific function (or all if func_name is '')."""
        cmd = [
            "python",
            f"{PRO_PATH}/r2_extract.py",
            self.binary_path,
            self.tmpfile,
            func_name
        ]
        # Only run IDA if tmpfile does not exist
        if not os.path.exists(self.tmpfile) or OVERWRITE == True:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                print('Error extracting features from exec file')
                print(f'cmd: {cmd}')
                print(f'return code: {process.returncode}')
                print(f'output stream: {process.stdout}')
                print(f'error stream: {process.stderr}')
                return None
        features = list(self._read_features())
        # self._delete_tmpfile()  # to remove tmpfile after reading
        return features


def is_static_archive(filename):
    """Check if the file is a static archive (.a)."""
    return filename.lower().endswith('.a')

def extract_object_files(archive_path, extract_dir):
    """Extract all .o files from a static archive using 'ar'."""
    archive_path_abs = os.path.abspath(archive_path)
    cmd = f'ar x "{archive_path_abs}"'
    process = subprocess.run(cmd, shell=True, cwd=extract_dir, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Failed to extract object files from {archive_path}")
        print(process.stderr)
        return []
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.o')]


def extract_features(args):
    binary_path = args.binaryfile
    func_name = args.f if args.f else ''
    out_file = args.o if args.o else ''
    func_dicts = []

    print(f"starting feature extraction for: {binary_path}")
    if func_name:
        print(f"target function: {func_name}")
    else:
        print("extracting features for all functions.")

    def process_feature_dict(dic):
        """Convert raw feature dict to standardized format."""
        nodes_ordered = [node for node in dic.keys() if str(node).startswith('0x')]
        feature_list = []
        adj_matrix = [[0 for _ in nodes_ordered] for _ in nodes_ordered]
        for i, node in enumerate(nodes_ordered):
            feature_list.append([
                len(dic[node]["String_Constant"]),
                len(dic[node]["Numberic_Constant"]),
                dic[node]["No_Tran"],
                dic[node]["No_Call"],
                dic[node]["No_Instru"],
                dic[node]["No_Arith"],
                dic[node]["No_offspring"],
            ])
            for presuccessor in dic[node]['pre']:
                p_i = nodes_ordered.index(presuccessor)
                adj_matrix[p_i][i] = 1
        print(f"[DEBUG] Processed function: {dic['fun_name']} with {len(nodes_ordered)} nodes")
        return {
            "func_name": dic['fun_name'],
            "feature_list": feature_list,
            "adjacent_matrix": adj_matrix
        }

    if is_static_archive(binary_path):
        print("detected static archive (.a). extracting object files")
        # Handle static archive: extract all .o files and process each
        with tempfile.TemporaryDirectory() as tmpdir:
            object_files = extract_object_files(binary_path, tmpdir)
            print(f"found {len(object_files)} object files")
            if not object_files:
                print("no object files found in archive")
                return
            for obj_file in object_files:
                extractor = FeatureExtractor(obj_file)
                features = extractor.extract_function(func_name)
                if features:
                    for dic in features:
                        func_dicts.append(process_feature_dict(dic))
                else:
                    print(f"no features extracted from {obj_file}")
    else:
        extractor = FeatureExtractor(binary_path)
        features = extractor.extract_function(func_name)
        if features:
            print(f"extracted features from {binary_path}")
            for dic in features:
                func_dicts.append(process_feature_dict(dic))
        else:
            print(f"no features extracted from {binary_path}")

    # output results to json
    if out_file:
        with open(out_file, 'w') as f:
            json.dump(func_dicts, f, separators=(',', ":"))
        print("[INFO] Feature extraction complete")
    else:
        for x in func_dicts:
            print(x)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('binaryfile', help='file to be analysed')
    parse.add_argument('-f', help='function name to be handled ')
    parse.add_argument('-b', help='if file is ida db (potential feat for ida driver?)', action='store_true', default=False)
    parse.add_argument('-o', help='output filename')
    args = parse.parse_args()
    extract_features(args)