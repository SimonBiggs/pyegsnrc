import re

subs = {
    r'"(.*?)"': r'# \1',  # comments in quotes
    r'"(.*?)$': r'# \1',  # comment without end quote
}

if __name__ == "__main__":
    in_filename = "electr.mor"
    out_filename = "electr.py"
    with open(in_filename, 'r') as f:
        code = f.read()

    for pattern, sub in subs.items():
        code = re.sub(pattern, sub, code)

    with open(out_filename, "w") as f:
        f.write(code)

