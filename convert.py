import re

import sys

from typing import List


if sys.version_info < (3, 8):
    raise ImportError("Need Python 3.8 or later")

# XXX needs to be OrderedDict!!
main_subs = {
    # Comments / semicolons
    r'"(.*?)"': r'# \1',  # comments in quotes
    r'"(.*)': r'# \1',  # comment without end quote
    r";(\s*)$": r"\1",      # semi-colon at end of line
    r";(\s*)(?P<comment>#(.*?))?$": r" \g<comment>", # still a semicolon before #

    # IF/ELSE
    r"^(\s*)IF\((.*)\)\s*\[(.*?)[;]?\](.*)$": r"\1if \2:\n\1    \3\4", # basic IF
    r"^(\s*)(?:]\s*)?ELSE(.*)\[(.*)\](.*)$": r"\1else:\n\1    \3\4", # basic ELSE [ ]
    r"^(\s*)(?:]\s*)?ELSE(\s*)\[?.*?$": r"\1else:",  # bare ELSE line or ELSE [
    r"^(\s*)IF(\s*)?\((.*)\)(.*)$": r"\1if \3:\n\1    \4", # IF on one line

    # LOOPs
    r"^(\s*):(\w*):LOOP": r"\1while True:  # :\2: LOOP",
    r"^(\s*)\]\s*UNTIL\s*\((.*?)\)(\s*?# .*$)?": r"\1if \2:\n\1    break \3",


    # Math operators
    r"if(.*?)~=": r"if\1!=", # not equals
    r"if(.*?) = ": r"if\1 == ", # = to ==
    r"if(.*?) = ": r"if\1 == ", # = to == again if there multiple times
    r"if(.*?) = ": r"if\1 == ", # = to == again

    # Booleans
    r" \| ": r" or ",
    r" \& ": r" and ",

    # Leftover brackets
    r"^\s*\[\s*$": r"",  # line with only [
    r"^\s*\]\s*$": r"",  # line with only ]


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
    r"\$IMPLICIT-NONE": r"",
    r"\$DEFINE-LOCAL-VARIABLES-ELECTR": r"# $DEFINE-LOCAL-VARIABLES-ELECTR XXX do we need to type these?",


}

call_subs= {
    # paired with added funcs
    r"\$start_new_particle": r"start_new_particle()",
    r"\$CALL_USER_ELECTRON": r"call_user_electron()",
    r"\$SELECT_ELECTRON_MFP": r"select_electron_mfp()",
    r"   \$RANDOMSET (\w*)": r"\1 = randomset()",
}


def add_new_funcs(code: str) -> str:
    fakes = [
        "def start_new_particle():\n    medium = med[irl]\n\n",
        "def call_user_electron():\n    pass\n\n",
        "def select_electron_mfp():\n    RNNE1 = randomset()\n    if RNNE1 == 0.0):\n        RNNE1 = 1.E-30\n    DEMFP = max([-log(RNNE1), EPSEMFP])",
    ]
    return "\n".join(fakes) + code

commenting_lines = [
    "/******* trying to save evaluation of range.",
    "*/",
    "data ierust/0/ # To count negative ustep's",
    "save ierust",

]

def replace_subs(code: str, subs: dict) -> str:
    for pattern, sub in subs.items():
        code = re.sub(pattern, sub, code, flags=re.MULTILINE)
    return code


def replace_var_decl(code: str) -> str:
    mapping = {
        "$INTEGER": "int",
        "$REAL": "float",
        "$LOGICAL": "bool",
        "LOGICAL": "bool",
    }

    out_lines = []
    for line in code.splitlines():
        matched = False
        for typ in ["$INTEGER", "$REAL", "$LOGICAL", "LOGICAL"]:
            if line.startswith(typ):
                vars = line.replace(typ, "").split(",")
                for var in vars:
                    out_lines.append(f"{var.strip()}: {mapping[typ]}")
                matched = True
                break # out of inner loop
        if not matched:
            out_lines.append(line)

    return "\n".join(out_lines)

def comment_out_lines(code: str, lines_to_comment: list) -> str:
    all_lines = code.splitlines()
    for i, line in enumerate(all_lines):
        if line in lines_to_comment:
            all_lines[i] = "# " + line
    return "\n".join(all_lines)


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



def arrays_to_square_bracket(code: str) -> str:
    """Replace arrays with () in Mortran to [] in Python"""
    array_names = "ir med ecut WT iq e x y z u v w".split()
    for name in array_names:
        pattern = rf"([^\w]){name}\((.*?)\)"
        subst = rf"\1{name}[\2]"
        code = re.sub(pattern, subst, code)
    return code

if __name__ == "__main__":
    in_filename = "electr.mortran"
    out_filename = "electr.py"

    # in_filename = "egsnrc.macros"
    # out_filename = "common.py"

    with open(in_filename, 'r') as f:
        code = f.read()

    code = replace_subs(code, main_subs)
    code = fix_identifiers(code)
    # code = transpile_macros(code)
    code = replace_auscall(code)
    code = add_new_funcs(code)
    code = replace_subs(code, call_subs)
    code = replace_var_decl(code)
    code = comment_out_lines(code, commenting_lines)
    code = arrays_to_square_bracket(code)

    with open(out_filename, "w") as f:
        f.write(code)

