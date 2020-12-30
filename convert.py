import re

# XXX needs to be OrderedDict!!
subs = {
    r'"(.*?)"': r'# \1',  # comments in quotes
    r'"(.*)': r'# \1',  # comment without end quote
    r";(\s*)$": r"\1",      # semi-colon at end of line
    r";(\s*)(?P<comment>#(.*?))?$": r" \g<comment>",
}

if __name__ == "__main__":
    in_filename = "electr.mor"
    out_filename = "electr.py"
    with open(in_filename, 'r') as f:
        code = f.read()

    for pattern, sub in subs.items():
        code = re.sub(pattern, sub, code, flags=re.MULTILINE)

    with open(out_filename, "w") as f:
        f.write(code)

