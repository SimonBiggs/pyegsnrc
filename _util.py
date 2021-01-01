"""Various routines for analyzing what are calls, what are constants, etc."""
import re

# def evaluate_using(P1, P2, P3):
#     if P2 == SNAME1:
#         P1=P2}1(L{P3})*{P3}+{P2}0(L{P3})
#     else:
#         {P1}={P2}1(L{P3},MEDIUM)*{P3}+{P2}0(L{P3},MEDIUM)

def test_eval_subst(code):
    pattern = r"\$EVALUATE (\w*) USING (\w*)\((\w*)\);?"
    subst = """
    [IF] '\g<2>'=SNAME1
    [\g<1>=\g<2>1(L\g<3>)*\g<3>+\g<2>0(L\g<3>);] [ELSE]
    [\g<1>=\g<2>1(L\g<3>,MEDIUM)*\g<3>+\g<2>0(L\g<3>,MEDIUM);]}
    """
    # subst = r"\1"
    # m = re.search(pattern, code)
    # print(m.groups())

    code = re.sub(pattern, subst, code, re.MULTILINE)
    return code


# REPLACE {$EVALUATE#USING#(#);} WITH {
#   [IF] '{P2}'=SNAME1
#   [{P1}={P2}1(L{P3})*{P3}+{P2}0(L{P3});] [ELSE]
#   [{P1}={P2}1(L{P3},MEDIUM)*{P3}+{P2}0(L{P3},MEDIUM);]}
# "{P1} IS VARIABLE TO BE ASSIGNED VALUE."
# "{P2} IS THE FUNCTION BEING APPROXIMATED."
# "{P3} IS THE ARGUMENT OF THE FUNCTION. IN THE CURRENT"
# "PWLF METHOD, THE ARGUMENT DETERMINES AN INTERVAL USING THE"
# "$SET INTERVAL MACROS.   WITH IN THIS INTERVAL THE"
# "FUNCTION IS APPROXIMATED AS A LINEAR FUNCTION OF"
# "THE ARGUMENT. BUT"
# "IF {P2}=SIN IT DOES NOT DEPEND ON MEDIUM"

# REPLACE {$EVALUATE#USING#(#,#);} WITH {
#   {P1}={P2}0(L{P3},L{P4})+{P2}1(L{P3},L{P4})*{P3}+
#   {P2}2(L{P3},L{P4})*
#   {P4};}"2-D APPROXIMATION INDEPENDENT OF MEDIUM"
# SPECIFY SNAME AS ['sinc'|'blc'|'rthr'|'rthri'|'SINC'|'BLC'|'RTHR'|'RTHRI'];
# SPECIFY SNAME1 AS ['sin'|'SIN'];


def find_all_macros_used(code):
    """Return all identifiers starting with $ in the code"""
    pattern = r" *?(\$[\w-]*)" #r"^ *?(\$[-\w]*)"
    matches = re.findall(pattern, code)
    return set(matches)


def find_macros_including_macros(code):
    """Matches where the WITH replace also has $ in it"""
    # pattern = r"REPLACE\s*?\{\s*?(\$[\w-]*);?\}\s*?WITH\s*?\{(.*\$.*);?\}"
    # matches = [m for m in re.finditer(pattern, code, flags=re.MULTILINE)]
    # return [m.groups() for m in matches]

    return {
        k:v for k,v in find_all_replaces(code).items()
        if '$' in v
    }

def find_all_replaces(code):
    pattern = r"REPLACE\s*?\{(.*)\}\s*?WITH\s*?\{(.*)\}"
    return dict(m.groups() for m in re.finditer(pattern, code))

if __name__ == "__main__":
    # test_code = "$EVALUATE dedx0 USING ededx(elke);"
    # print("Subst for ", test_code)
    # print(test_eval_subst(test_code))
    from pprint import pprint

    in_filename = "electr.mortran"
    with open(in_filename, 'r') as f:
        code = f.read()

    with open("egsnrc.macros", 'r') as f:
        macros_code = f.read()


    # print(find_all_macros_used(code))
    # print("Macros within macros")
    pprint(find_macros_including_macros(macros_code))
    # print(results)

    # print(find_all_replaces(macros_code))