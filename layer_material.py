import sys

input_file = sys.argv[1]
with open(input_file, 'r') as f:
    lines = f.readlines()

layer = 0
with open(input_file, 'w') as f:
    for line in lines:
        if line.startswith(';LAYER_CHANGE'):
            layer += 1
            tool = 0 if layer % 2 == 1 else 1
            f.write(line)
            f.write(f'T{tool}\n')
            f.write(f'M620 S{tool}A\n')
        else:
            f.write(line)