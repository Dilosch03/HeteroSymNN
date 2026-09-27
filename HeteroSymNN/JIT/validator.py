"""
Static validation of user formulas before they reach SymPy.

``sympy.parse_expr`` evaluates its input with Python's ``eval``, and formulas can come from
untrusted places (``.symnn`` archives, the CLI). So every formula is first parsed with Python's
own ``ast`` module, which never executes anything, and accepted only if every node is plain math:

* numbers, ``True``/``False`` and ASCII names that don't start with ``_``;
* the operators ``+ - * / **``, unary ``+ -``, comparisons ``< <= > >=`` and ``& |``
  (the last two build conditions for ``Piecewise``);
* tuples (the ``(expression, condition)`` pairs of ``Piecewise``);
* calls to the math functions in :data:`MATH_FUNCTIONS` or to a formula alias, with positional
  arguments only. Calls can be nested (``sin(tanh(num))``).

Everything else (attribute access, subscripts, strings, lambdas, comprehensions, keyword
arguments, ...) is rejected. Size limits also guard against formulas built to exhaust memory
or CPU while SymPy simplifies or differentiates them.
"""
import ast
import keyword
import math
import numbers
from dataclasses import dataclass

from ..exceptions import FormulaParsingError
from . import codegen

__all__ = ["validate_formula", "validate_constants", "FormulaNames", "MATH_FUNCTIONS", "ALLOWED_CALLS"]

# Functions whose value AND derivative translate to both the CUDA and the NumPy backends.
MATH_FUNCTIONS = frozenset({
    "sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "log", "sqrt", "cbrt", "Abs", "Max", "Min", "erf", "erfc",
    "Piecewise",
})
ALLOWED_CALLS = MATH_FUNCTIONS | frozenset(codegen.COMMON_FORMULAS)

MAX_FORMULA_LENGTH = 2000
MAX_NODES = 500
MAX_DEPTH = 32
MAX_CONSTANT_EXPONENT = 64

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.Tuple, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr,
    ast.UAdd, ast.USub,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)
_FRIENDLY_NAMES = {
    ast.Attribute: "attribute access ('.')", ast.Subscript: "indexing ('[...]')", ast.Lambda: "lambda functions",
    ast.List: "lists", ast.Dict: "dictionaries", ast.Set: "sets", ast.ListComp: "comprehensions",
    ast.GeneratorExp: "comprehensions", ast.DictComp: "comprehensions", ast.SetComp: "comprehensions",
    ast.IfExp: "'if/else' expressions (use Piecewise)", ast.BoolOp: "'and'/'or' (use '&' and '|')",
    ast.Mod: "the '%' operator", ast.FloorDiv: "the '//' operator", ast.Eq: "'==' (use Piecewise conditions)",
    ast.NotEq: "'!='", ast.Starred: "argument unpacking", ast.NamedExpr: "the ':=' operator",
}


@dataclass(frozen=True)
class FormulaNames:
    """Names found in a validated formula."""
    called: frozenset
    symbols: frozenset


def _is_valid_name(name) -> bool:
    return isinstance(name, str) and name.isascii() and name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_")


def _depth(node, level=1):
    children = list(ast.iter_child_nodes(node))
    return level if not children else max(_depth(child, level + 1) for child in children)


def _contains(node, node_type) -> bool:
    return any(isinstance(n, node_type) for n in ast.walk(node))


def validate_formula(formula) -> FormulaNames:
    """
    Checks that a formula string is plain math, without evaluating it.

    Parameters
    ----------
    formula : str
        The formula to validate, e.g. ``"tanh(num) * a"``.

    Returns
    -------
    FormulaNames
        The called function names and the free symbol names (variables and constants).

    Raises
    ------
    :exc:`~HeteroSymNN.exceptions.FormulaParsingError`
        If the formula contains anything outside the allowed grammar or exceeds the size limits.
    """
    if not isinstance(formula, str):
        raise FormulaParsingError(f"Formulas must be strings, got {type(formula).__name__}.")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise FormulaParsingError(f"Formula is too long ({len(formula)} characters, maximum {MAX_FORMULA_LENGTH}).")

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as e:
        raise FormulaParsingError(f"Invalid mathematical syntax in '{formula}'. Please check your operators and parentheses.") from None
    except (RecursionError, MemoryError, ValueError):
        raise FormulaParsingError(f"Formula '{formula[:80]}' could not be parsed (too deeply nested or contains invalid characters).") from None

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise FormulaParsingError(f"Formula '{formula[:80]}...' is too complex ({len(nodes)} elements, maximum {MAX_NODES}).")
    if _depth(tree) > MAX_DEPTH:
        raise FormulaParsingError(f"Formula '{formula[:80]}' is nested too deeply (maximum depth {MAX_DEPTH}).")

    # Name nodes that are the target of a call are functions, every other name is a symbol (the Golden Rule).
    call_targets = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    called, symbols = set(), set()
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            what = _FRIENDLY_NAMES.get(type(node), f"'{type(node).__name__}' expressions")
            raise FormulaParsingError(f"Formula '{formula}' uses {what}, which is not allowed in a math formula.")

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (bool, int, float)):
                raise FormulaParsingError(f"Formula '{formula}' contains the literal {node.value!r}; only real numbers are allowed.")

        elif isinstance(node, ast.Name):
            if not _is_valid_name(node.id):
                raise FormulaParsingError(f"'{node.id}' in formula '{formula}' is not a valid name (use ASCII letters, digits and '_', not starting with '_').")
            if id(node) not in call_targets:
                symbols.add(node.id)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaParsingError(f"Formula '{formula}' calls something that is not a plain function name.")
            if node.keywords:
                raise FormulaParsingError(f"Formula '{formula}' passes keyword arguments to '{node.func.id}'; only positional arguments are allowed.")
            if node.func.id not in ALLOWED_CALLS:
                raise FormulaParsingError(
                    f"'{node.func.id}' in formula '{formula}' is not a supported function. "
                    f"Supported: {', '.join(sorted(ALLOWED_CALLS))}. "
                    f"If '{node.func.id}' is meant to be a constant, remove the parentheses."
                )
            called.add(node.func.id)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and not _contains(node, ast.Name):
            # Constant-only powers are evaluated exactly by SymPy: '9**9**9' would never finish.
            exponent = node.right
            if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(exponent)):
                raise FormulaParsingError(f"Formula '{formula}' contains a tower of constant powers, which is not allowed.")
            for literal in (n.value for n in ast.walk(exponent) if isinstance(n, ast.Constant)):
                if not math.isfinite(literal) or abs(literal) > MAX_CONSTANT_EXPONENT:
                    raise FormulaParsingError(f"Formula '{formula}' raises a constant to the power {literal} (maximum {MAX_CONSTANT_EXPONENT}).")

    return FormulaNames(called=frozenset(called), symbols=frozenset(symbols))


def validate_constants(constants) -> None:
    """
    Checks the constants dictionary of a node: names must be valid identifiers and values real numbers.

    Raises
    ------
    :exc:`~HeteroSymNN.exceptions.FormulaParsingError`
        If a name or value is invalid.
    """
    if constants is None:
        return
    if not isinstance(constants, dict):
        raise FormulaParsingError(f"Constants must be a dictionary of name -> number, got {type(constants).__name__}.")
    for name, value in constants.items():
        if not _is_valid_name(name):
            raise FormulaParsingError(f"Constant name {name!r} is not valid (use ASCII letters, digits and '_', not starting with '_').")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise FormulaParsingError(f"Constant '{name}' must be a real number, got {type(value).__name__}.")
