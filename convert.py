import re

import sys

from typing import List


if sys.version_info < (3, 8):
    raise ImportError("Need Python 3.8 or later")

# XXX needs to be OrderedDict!!
subs = {
    r'"(.*?)"': r'# \1',  # comments in quotes
    r'"(.*)': r'# \1',  # comment without end quote
    r";(\s*)$": r"\1",      # semi-colon at end of line
    r";(\s*)(?P<comment>#(.*?))?$": r" \g<comment>", # still a semicolon before #
    r"^(\s*)IF\((.*)\)\s*\[(.*?)[;]?\](.*)$": r"\1if \2:\n\1    \3\4", # basic IF
    r"^(\s*)(?:]\s*)?ELSE(.*)\[(.*)\](.*)$": r"\1else:\n\1    \3\4", # basic ELSE
    r"^(\s*)(?:]\s*)?ELSE(\s*)$": r"\1else:",  # bare ELSE line
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
    r"\$DECLARE-PEGS4-COMMON-BLOCKS": "",
    r"SUBROUTINE\s*(.*)$": r"def \1:",
    r"\.true\.": "True",
    r"\.false\.": "False",
    r"[iI]f(.*?)(?:=)?=\s*True": r"if\1 is True",
    r"[iI]f(.*?)(?:=)?=\s*False": r"if\1 is False",
}


def replace_subs(code: str) -> str:
    for pattern, sub in subs.items():
        code = re.sub(pattern, sub, code, flags=re.MULTILINE)
    return code


def fix_identifiers(code) -> str:
    """Take invalid (for Python) var names and make them valid"""
    # Fix leading number, often used for ranges
    code = re.sub(r"^[^#].*\$(\d)", r"\$from\1", code)

    # Fix dashes to underscore
    # First, some comment have "---", keep those
    code = re.sub(r'"(.*)\$(\w*)---', r'"\1$\2 --- ', code)
    for i in range(8, 1, -1):
        pattern = r"\$" + "-".join([r"(\w*)"]*i)
        subst = r"$" + "_".join([rf"\{j}" for j in range(1, i+1)])
        code = re.sub(pattern, subst, code)
    return code


def transpile_macros(code: str) -> str:
    """Transpile statements in macros file"""
    macro_subs = {
        # PARAMETER with comments
        r'PARAMETER\s*\$(\w*)\s*=\s*(\d*);\s*"(.*?)"': r"\1: int = \2  # \3",
        r'PARAMETER\s*\$(\w*)\s*=\s*(\d*\.\d*);\s*"(.*?)"': r"\1: float = \2  # \3",

        # PARAMETER without comments
        r'PARAMETER\s*\$(\w*)\s*=\s*(\d*);': r"\1: int = \2",
        r'PARAMETER\s*\$(\w*)\s*=\s*(\d*\.\d*);': r"\1: float = \2",

        # REPLACE
        r'REPLACE\s*\{\$(\w*)\}\s*WITH\s*\{(\d*)\}': r"\1: int = \2",  # simple int replacement
        r'REPLACE\s*\{\$(\w*)\}\s*WITH\s*\{(\d\.?\d*)\}': r"\1: float = \2",  # simple float replace

        # Comments, semicolon
        r'"(.*?)"': r'# \1',  # comments in quotes
        r'"(.*)': r'# \1',  # comment without end quote
        r";(\s*)$": r"\1",      # semi-colon at end of line
        r";\s*(# .*$)?": r"\1", # '; # comment' -> just comment

        # Compiler directives
        r"^%(.*?)$": r"# %\1",  # Any line starting with %

    }
    for pattern, sub in macro_subs.items():
        code = re.sub(pattern, sub, code, flags=re.MULTILINE)

    return code


def replace_auscall(code: str) -> str:
    """Return list of strings, indented properly

    XXX ASSUMES
    - AUSGAB is first on line, other than whitespace
    - no comments are anything else on line

    """
    # pattern = r"^(?P<indent>\s*)\$AUSCALL\((?P<arg>.*?)\)\s*?;?.*?$"
    pattern = r"^( *?)\$AUSCALL\((.*)\)"
    subst = (
        r"\1IARG = \2\n"
        r"\1if IAUSFL[IARG + 1] != 0:\n"
        r"\1    AUSGAB(IARG)"
    )
    # subst = "XXXXYYYY"
    code = re.sub(pattern, subst, code, flags=re.MULTILINE)
    fake_ausgab = """\n\ndef AUSGAB(IARG):\n    pass\n\n\n"""

    return fake_ausgab + code


if __name__ == "__main__":
    in_filename = "electr.mor"
    out_filename = "electr.py"

    # in_filename = "egsnrc_macros.mor"
    # out_filename = "common.py"

    with open(in_filename, 'r') as f:
        code = f.read()

    code = replace_subs(code)
    # code = fix_identifiers(code)
    # code = transpile_macros(code)
    code = replace_auscall(code)

    with open(out_filename, "w") as f:
        f.write(code)

