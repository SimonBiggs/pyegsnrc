import re

import sys


if sys.version_info < (3, 8):
    raise ImportError("Need Python 3.8 or later")

# XXX needs to be OrderedDict!!
subs = {
    r'"(.*?)"': r'# \1',  # comments in quotes
    r'"(.*)': r'# \1',  # comment without end quote
    r";(\s*)$": r"\1",      # semi-colon at end of line
    r";(\s*)(?P<comment>#(.*?))?$": r" \g<comment>",
    r"^(\s*)IF\((.*)\)\s*\[(.*?)[;]?\](.*)$": r"\1if \2:\n\1    \3\4", # basic IF
    r"^(\s*)ELSE(.*)\[(.*)\](.*)$": r"\1else:\n\1    \3\4", # basic ELSE
    r"^(\s*)ELSE(\s*)$": r"\1else:",  # bare ELSE line
    r"^(\s*)IF(\s*)?\((.*)\)(.*)$": r"\1if \3:\n\1    \4", # IF on one line
    r"if(.*?)~=": r"if\1!=", # not equals
    r"if(.*?) = ": r"if\1 == ", # = to ==
    r"if(.*?) = ": r"if\1 == ", # = to == again if there multiple times
    r"if(.*?) = ": r"if\1 == ", # = to == again
    r" \| ": r" or ",
    r" \& ": r" and ",
    r"^\s*\[\s*$": r"",  # line with only [
    r"^\s*\]\s*$": r"",  # line with only ]
    r"\$start_new_particle": r"medium = med(irl)",
    r"\$electron_region_change|\$photon_region_change": r"ir(np) = irnew; irl = irnew; medium = med(irl)",
    r"\$declare_max_medium": r"",
    r"\$default_nmed": "1",
    r"\$INIT-PEGS4-VARIABLES": "",
    r"\$DECLARE-PEGS4-COMMON-BLOCKS": ""
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

