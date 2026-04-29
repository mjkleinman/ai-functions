"""Scipy Backpropagation Demo — Self-contained DS-1000 Example.

Demonstrates AI function backpropagation using two deterministic improvements
from the DS-1000 Scipy benchmark (default mode, temperature=0 — same result
every run):

  scipy_787: scipy.optimize.minimize — Direct generation assigns raw OptimizeResult to `out`
             (TypeError). Directly memory-driven: training on scipy_716 teaches
             "objective takes single array; close over args; access result.x"

Three steps:
  1. Direct test   — run both test problems with empty memory, show code + error
  2. Training      — run 8 training problems in parallel, accumulate memory via
                     backprop, display the learned memory
  3. Trained test  — re-run both test problems with trained memory, show result

The explain_error ai_function provides a plain-English summary of each failure.

Usage:
    cd examples
    conda run -n aifunciton python scipy_backprop_demo.py
"""

import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from strands.models import BedrockModel

from ai_functions import ai_function
from ai_functions.memory.json_backend import JSONMemoryBackend
from ai_functions.optimizer.textgrad import TextGradOptimizer
from ai_functions.utils import bullet_points

from ds1000_utils import recall_params, run_problem, run_batch_parallel, build_feedback

console = Console()


# ---------------------------------------------------------------------------
# Inline DS-1000 problem data (no HuggingFace / benchmarks_public import)
# ---------------------------------------------------------------------------

# 8 training problems: scipy_711 through scipy_718
# Topics: log regression (polyfit), curve_fit, KS test, minimize API, norm.cdf
# scipy_716 is the key example — teaches the correct minimize() contract
TRAIN_PROBLEMS = [
    {
        "id": "scipy_711",
        "library": "Scipy",
        "prompt": 'Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). \nHow do I fit y = Alogx + B using polyfit()? The result should be an np.array of [A, B]\nA:\n<code>\nimport numpy as np\nimport scipy\nx = np.array([1, 7, 20, 50, 79])\ny = np.array([10, 19, 30, 35, 51])\n\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n',
        "code_context": 'import numpy as np\nimport copy\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            x = np.array([1, 7, 20, 50, 79])\n            y = np.array([10, 19, 30, 35, 51])\n        return x, y\n\n    def generate_ans(data):\n        _a = data\n        x, y = _a\n        result = np.polyfit(np.log(x), y, 1)\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy\nx, y = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_712",
        "library": "Scipy",
        "prompt": 'Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). \nHow do I fit y = A + Blogx using polyfit()? The result should be an np.array of [A, B]\nA:\n<code>\nimport numpy as np\nimport scipy\nx = np.array([1, 7, 20, 50, 79])\ny = np.array([10, 19, 30, 35, 51])\n\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n',
        "code_context": 'import numpy as np\nimport copy\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            x = np.array([1, 7, 20, 50, 79])\n            y = np.array([10, 19, 30, 35, 51])\n        return x, y\n\n    def generate_ans(data):\n        _a = data\n        x, y = _a\n        result = np.polyfit(np.log(x), y, 1)[::-1]\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy\nx, y = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_713",
        "library": "Scipy",
        "prompt": 'Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). But I found no such functions for exponential and logarithmic fitting.\nHow do I fit y = A*exp(Bx) + C ? The result should be an np.array of [A, B, C]. I know that polyfit performs bad for this function, so I would like to use curve_fit to solve the problem, and it should start from initial guess p0.\nA:\n<code>\nimport numpy as np\nimport scipy.optimize\ny = np.array([1, 7, 20, 50, 79])\nx = np.array([10, 19, 30, 35, 51])\np0 = (4, 0.1, 1)\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n',
        "code_context": 'import numpy as np\nimport copy\nimport scipy.optimize\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            y = np.array([1, 7, 20, 50, 79])\n            x = np.array([10, 19, 30, 35, 51])\n            p0 = (4, 0.1, 1)\n        return x, y, p0\n\n    def generate_ans(data):\n        _a = data\n        x, y, p0 = _a\n        result = scipy.optimize.curve_fit(\n            lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=p0\n        )[0]\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy.optimize\nx, y, p0 = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_714",
        "library": "Scipy",
        "prompt": "Problem:\nI can't figure out how to do a Two-sample KS test in Scipy.\nAfter reading the documentation scipy kstest\nI can see how to test where a distribution is identical to standard normal distribution\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\ntest_stat = kstest(x, 'norm')\n#>>> test_stat\n#(0.021080234718821145, 0.76584491300591395)\nWhich means that at p-value of 0.76 we can not reject the null hypothesis that the two distributions are identical.\nHowever, I want to compare two distributions and see if I can reject the null hypothesis that they are identical, something like:\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\nz = np.random.normal(1.1,0.9, 1000)\nand test whether x and z are identical\nI tried the naive:\ntest_stat = kstest(x, z)\nand got the following error:\nTypeError: 'numpy.ndarray' object is not callable\nIs there a way to do a two-sample KS test in Python? If so, how should I do it?\nThank You in Advance\nA:\n<code>\nfrom scipy import stats\nimport numpy as np\nnp.random.seed(42)\nx = np.random.normal(0, 1, 1000)\ny = np.random.normal(0, 1, 1000)\n</code>\nstatistic, p_value = ... # put solution in these variables\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nfrom scipy import stats\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(0, 1, 1000)\n        elif test_case_id == 2:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(1.1, 0.9, 1000)\n        return x, y\n\n    def generate_ans(data):\n        _a = data\n        x, y = _a\n        statistic, p_value = stats.ks_2samp(x, y)\n        return [statistic, p_value]\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nfrom scipy import stats\nimport numpy as np\nnp.random.seed(42)\nx, y = test_input\n[insert]\nresult = [statistic, p_value]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(2):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_715",
        "library": "Scipy",
        "prompt": "Problem:\nI can't figure out how to do a Two-sample KS test in Scipy.\nAfter reading the documentation scipy kstest\nI can see how to test where a distribution is identical to standard normal distribution\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\ntest_stat = kstest(x, 'norm')\n#>>> test_stat\n#(0.021080234718821145, 0.76584491300591395)\nWhich means that at p-value of 0.76 we can not reject the null hypothesis that the two distributions are identical.\nHowever, I want to compare two distributions and see if I can reject the null hypothesis that they are identical, something like:\nfrom scipy.stats import kstest\nimport numpy as np\nx = np.random.normal(0,1,1000)\nz = np.random.normal(1.1,0.9, 1000)\nand test whether x and z are identical\nI tried the naive:\ntest_stat = kstest(x, z)\nand got the following error:\nTypeError: 'numpy.ndarray' object is not callable\nIs there a way to do a two-sample KS test in Python, then test whether I can reject the null hypothesis that the two distributions are identical(result=True means able to reject, and the vice versa) based on alpha? If so, how should I do it?\nThank You in Advance\nA:\n<code>\nfrom scipy import stats\nimport numpy as np\nnp.random.seed(42)\nx = np.random.normal(0, 1, 1000)\ny = np.random.normal(0, 1, 1000)\nalpha = 0.01\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nfrom scipy import stats\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(0, 1, 1000)\n        elif test_case_id == 2:\n            np.random.seed(42)\n            x = np.random.normal(0, 1, 1000)\n            y = np.random.normal(1.1, 0.9, 1000)\n        alpha = 0.01\n        return x, y, alpha\n\n    def generate_ans(data):\n        _a = data\n        x, y, alpha = _a\n        s, p = stats.ks_2samp(x, y)\n        result = p <= alpha\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_array_equal(result, ans)\n    return 1\n\n\nexec_context = r"""\nfrom scipy import stats\nimport numpy as np\nx, y, alpha = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(2):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_716",  # KEY training example — teaches correct minimize() contract
        "library": "Scipy",
        "prompt": "Problem:\nAccording to the SciPy documentation it is possible to minimize functions with multiple variables, yet it doesn't tell how to optimize on such functions.\nfrom scipy.optimize import minimize\nfrom math import sqrt, sin, pi, cos\ndef f(c):\n  return sqrt((sin(pi/2) + sin(0) + sin(c) - 2)**2 + (cos(pi/2) + cos(0) + cos(c) - 1)**2)\nprint minimize(f, 3.14/2 + 3.14/7)\n\nThe above code does try to minimize the function f, but for my task I need to minimize with respect to three variables, starting from `initial_guess`.\nSimply introducing a second argument and adjusting minimize accordingly yields an error (TypeError: f() takes exactly 2 arguments (1 given)).\nHow does minimize work when minimizing with multiple variables.\nI need to minimize f(a,b,c)=((a+b-c)-2)**2 + ((3*a-b-c))**2 + sin(b) + cos(b) + 4.\nResult should be a list=[a,b,c], the parameters of minimized function.\n\nA:\n<code>\nimport scipy.optimize as optimize\nfrom math import sqrt, sin, pi, cos\n\ninitial_guess = [-1, 0, -3]\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nfrom scipy import optimize\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            a = [-1, 0, -3]\n        return a\n\n    def generate_ans(data):\n        _a = data\n        initial_guess = _a\n\n        def g(params):\n            a, b, c = params\n            return (\n                ((a + b - c) - 2) ** 2\n                + ((3 * a - b - c)) ** 2\n                + np.sin(b)\n                + np.cos(b)\n                + 4\n            )\n\n        res = optimize.minimize(g, initial_guess)\n        result = res.x\n        return result\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    def g(params):\n        a, b, c = params\n        return (\n            ((a + b - c) - 2) ** 2 + ((3 * a - b - c)) ** 2 + np.sin(b) + np.cos(b) + 4\n        )\n\n    assert abs(g(result) - g(ans)) < 1e-2\n    return 1\n\n\nexec_context = r"""\nimport scipy.optimize as optimize\nfrom math import sqrt, sin, pi, cos\ninitial_guess = test_input\n[insert]\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_717",
        "library": "Scipy",
        "prompt": "Problem:\nHow does one convert a list of Z-scores from the Z-distribution (standard normal distribution, Gaussian distribution) to left-tailed p-values? I have yet to find the magical function in Scipy's stats module to do this, but one must be there.\nA:\n<code>\nimport numpy as np\nimport scipy.stats\nz_scores = np.array([-3, -2, 0, 2, 2.5])\n</code>\np_values = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nimport scipy.stats\n\n\ndef generate_test_case(test_case_id):\n\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            a = np.array([-3, -2, 0, 2, 2.5])\n        return a\n\n    def generate_ans(data):\n        _a = data\n        z_scores = _a\n        temp = np.array(z_scores)\n        p_values = scipy.stats.norm.cdf(temp)\n        return p_values\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy.stats\nz_scores = test_input\n[insert]\nresult = p_values\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
    {
        "id": "scipy_718",
        "library": "Scipy",
        "prompt": "Problem:\nHow does one convert a list of Z-scores from the Z-distribution (standard normal distribution, Gaussian distribution) to left-tailed p-values? Original data is sampled from X ~ N(mu, sigma). I have yet to find the magical function in Scipy's stats module to do this, but one must be there.\nA:\n<code>\nimport scipy.stats\nimport numpy as np\nz_scores = [-3, -2, 0, 2, 2.5]\nmu = 3\nsigma = 4\n</code>\np_values = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport pandas as pd\nimport scipy\nfrom scipy import sparse\nimport scipy.stats\nimport copy\nimport io\nfrom scipy import integrate\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            z_scores = [-3, -2, 0, 2, 2.5]\n            mu = 3\n            sigma = 4\n        return z_scores, mu, sigma\n\n    def generate_ans(data):\n        _a = data\n        z_scores, mu, sigma = _a\n        temp = np.array(z_scores)\n        p_values = scipy.stats.norm.cdf(temp)\n        return p_values\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    np.testing.assert_allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport numpy as np\nimport scipy.stats\nz_scores, mu, sigma = test_input\n[insert]\nresult = p_values\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },
]

# 1 test problem: scipy_787
TEST_PROBLEMS = [
    {
        "id": "scipy_787",
        "library": "Scipy",
        "prompt": "Problem:\n\n\nI am having a problem with minimization procedure. Actually, I could not create a correct objective function for my problem.\nProblem definition\n•\tMy function: yn = a_11*x1**2 + a_12*x2**2 + ... + a_m*xn**2,where xn- unknowns, a_m - coefficients. n = 1..N, m = 1..M\n•\tIn my case, N=5 for x1,..,x5 and M=3 for y1, y2, y3.\nI need to find the optimum: x1, x2,...,x5 so that it can satisfy the y\nMy question:\n•\tHow to solve the question using scipy.optimize?\nMy code:   (tried in lmfit, but return errors. Therefore I would ask for scipy solution)\nimport numpy as np\nfrom lmfit import Parameters, minimize\ndef func(x,a):\n    return np.dot(a, x**2)\ndef residual(pars, a, y):\n    vals = pars.valuesdict()\n    x = vals['x']\n    model = func(x,a)\n    return (y - model)**2\ndef main():\n    # simple one: a(M,N) = a(3,5)\n    a = np.array([ [ 0, 0, 1, 1, 1 ],\n                   [ 1, 0, 1, 0, 1 ],\n                   [ 0, 1, 0, 1, 0 ] ])\n    # true values of x\n    x_true = np.array([10, 13, 5, 8, 40])\n    # data without noise\n    y = func(x_true,a)\n    #************************************\n    # Apriori x0\n    x0 = np.array([2, 3, 1, 4, 20])\n    fit_params = Parameters()\n    fit_params.add('x', value=x0)\n    out = minimize(residual, fit_params, args=(a, y))\n    print out\nif __name__ == '__main__':\nmain()\nResult should be optimal x array. The method I hope to use is L-BFGS-B, with added lower bounds on x.\n\nA:\n\n\n<code>\nimport scipy.optimize\nimport numpy as np\nnp.random.seed(42)\na = np.random.rand(3,5)\nx_true = np.array([10, 13, 5, 8, 40])\ny = a.dot(x_true ** 2)\nx0 = np.array([2, 3, 1, 4, 20])\nx_lower_bounds = x_true / 2\n</code>\nout = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
        "code_context": 'import numpy as np\nimport copy\nimport scipy.optimize\n\n\ndef generate_test_case(test_case_id):\n    def define_test_input(test_case_id):\n        if test_case_id == 1:\n            np.random.seed(42)\n            a = np.random.rand(3, 5)\n            x_true = np.array([10, 13, 5, 8, 40])\n            y = a.dot(x_true**2)\n            x0 = np.array([2, 3, 1, 4, 20])\n            x_bounds = x_true / 2\n        return a, x_true, y, x0, x_bounds\n\n    def generate_ans(data):\n        _a = data\n        a, x_true, y, x0, x_lower_bounds = _a\n\n        def residual_ans(x, a, y):\n            s = ((y - a.dot(x**2)) ** 2).sum()\n            return s\n\n        bounds = [[x, None] for x in x_lower_bounds]\n        out = scipy.optimize.minimize(\n            residual_ans, x0=x0, args=(a, y), method="L-BFGS-B", bounds=bounds\n        ).x\n        return out\n\n    test_input = define_test_input(test_case_id)\n    expected_result = generate_ans(copy.deepcopy(test_input))\n    return test_input, expected_result\n\n\ndef exec_test(result, ans):\n    assert np.allclose(result, ans)\n    return 1\n\n\nexec_context = r"""\nimport scipy.optimize\nimport numpy as np\na, x_true, y, x0, x_lower_bounds = test_input\n[insert]\nresult = out\n"""\n\n\ndef test_execution(solution: str):\n    code = exec_context.replace("[insert]", solution)\n    for i in range(1):\n        test_input, expected_result = generate_test_case(i + 1)\n        test_env = {"test_input": test_input}\n        exec(code, test_env)\n        assert exec_test(test_env["result"], expected_result)\n',
    },

]


# ---------------------------------------------------------------------------
# Memory schema
# ---------------------------------------------------------------------------

class LearningMemory(BaseModel):
    coding_patterns: str = Field(
        default="No learned patterns yet.",
        description=(
            "Concise bullet-point list (MAX 15 items) of general, reusable coding patterns "
            "and idioms for scipy/data science. Each bullet should be one sentence. "
            "Merge similar patterns into a single bullet. Do not include problem-specific details."
        ),
    )
    common_pitfalls: str = Field(
        default="No known pitfalls yet.",
        description=(
            "Concise bullet-point list (MAX 15 items) of common, reusable pitfalls "
            "and mistakes to avoid. Each bullet should be one sentence. "
            "Merge similar pitfalls into a single bullet. Do not include problem-specific details."
        ),
    )


# ---------------------------------------------------------------------------
# AI functions
# ---------------------------------------------------------------------------

_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # model_id="us.anthropic.claude-sonnet-4-6",
    temperature=0,  # deterministic — same result every run
)


@ai_function(model=_model, callback_handler=None)
def generate_code(
    coding_patterns: str,
    common_pitfalls: str,
    problem_prompt: str,
    library: str,
) -> str:
    """Solve the data science problem below by generating Python code.

    Output ONLY the Python code — no explanations, no markdown fences.
    The code will be inserted directly into an execution environment where
    pandas, numpy, and the input variables (df, a, x, List, etc.) are already defined.

    RULES:
    
    <learned_patterns>{coding_patterns}</learned_patterns>
    <pitfalls_to_avoid>{common_pitfalls}</pitfalls_to_avoid>

    <library>{library}</library>
    <problem>
    {problem_prompt}
    </problem>
    """


@ai_function(callback_handler=None)
def concise_consolidate(value: str, feedback: list[str]) -> str:
    """Update the following value by incorporating the feedback.

    <current_value>
    {value}
    </current_value>

    <feedback>
    {bullet_points(feedback)}
    </feedback>

    RULES FOR THE UPDATED VALUE:
    - Output a concise bullet-point list, one line per bullet, prefixed with "- "
    - Maximum 15 bullet points total
    - Each bullet must be a general, reusable pattern (not tied to a specific problem)
    - Merge bullets that say essentially the same thing into one
    - Keep each bullet to one sentence
    """


@ai_function(callback_handler=None)
def explain_error(
    problem_prompt: str,
    solution_code: str,
    error_message: str,
) -> str:
    """Explain in plain English why this code failed based on the error.

    <problem>
    {problem_prompt}
    </problem>

    <solution_code>
    {solution_code}
    </solution_code>

    <error>
    {error_message}
    </error>

    In 2-3 sentences:
    1. What the code does wrong (not just what the error message says)
    2. What the correct approach should be
    Do not write any code. Be specific about the scipy API involved.
    """


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def show_code(title: str, code: str, border: str = "cyan"):
    console.print(Panel(
        Syntax(code.strip(), "python", theme="monokai", word_wrap=True),
        title=title, border_style=border, expand=False,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print(Panel(
        "[bold]Scipy Backpropagation Demo[/]\n\n"
        "[bold]Deterministic[/] memory-driven improvement from DS-1000 Scipy\n"
        "(temperature=0 — produces the same result every run):\n\n"
        "  [cyan]scipy_787[/]  minimize() — Direct generation assigns raw OptimizeResult; forgets .x\n\n"
        "Fix is directly memory-driven: scipy_716 teaches the correct minimize()\n"
        "contract (single-array objective, close over args, access result via .x).\n\n"
        "Steps: (1) direct test  →  (2) train 8 examples in parallel  →  (3) test with memory",
        border_style="bold magenta",
    ))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        memory_path = f.name

    # =========================================================================
    # STEP 1: Direct test — empty memory
    # =========================================================================
    console.print(Rule("[bold yellow]Step 1 — Direct Test (no memory)[/]"))

    memory = JSONMemoryBackend(LearningMemory, "demo", path=memory_path, quiet=True)
    empty_params = recall_params(memory)

    console.print("\n[dim]Memory (empty):[/]")
    console.print(f"  coding_patterns: {empty_params[0].value}")
    console.print(f"  common_pitfalls: {empty_params[1].value}\n")

    direct_results = {}
    for problem in TEST_PROBLEMS:
        with console.status(f"  Testing {problem['id']} (no memory)..."):
            solution, exec_result, elapsed, _ = run_problem(problem, empty_params, generate_code)

        status = "[green]PASS[/]" if exec_result.passed else "[red]FAIL[/]"
        console.print(f"  {problem['id']}: {status}  ({elapsed:.1f}s)")
        show_code(f"{problem['id']} — generated code", solution,
                  border="green" if exec_result.passed else "red")

        if not exec_result.passed:
            error_detail = exec_result.error or ""
            if exec_result.expected_output:
                error_detail += f"\n  Expected: {exec_result.expected_output}"
            if exec_result.actual_output:
                error_detail += f"\n  Got:      {exec_result.actual_output}"
            with console.status("  Explaining error..."):
                explanation = explain_error(
                    problem_prompt=problem["prompt"],
                    solution_code=solution,
                    error_message=error_detail,
                )
            console.print(Panel(
                f"[bold red]Error:[/] {error_detail}\n\n{explanation}",
                title="Why it failed", border_style="yellow",
            ))

        direct_results[problem["id"]] = {
            "solution": solution,
            "passed": exec_result.passed,
            "error": exec_result.error,
        }

    memory.close()

    # =========================================================================
    # STEP 2: Training — 8 examples in parallel, backprop into memory
    # =========================================================================
    console.print(Rule("[bold green]Step 2 — Training (8 examples, parallel)[/]"))
    console.print(
        f"  Training on: {[p['id'] for p in TRAIN_PROBLEMS]}\n"
        f"  [dim]scipy_716 is the key example — teaches correct minimize() contract[/]\n"
    )

    memory = JSONMemoryBackend(LearningMemory, "demo", path=memory_path, quiet=True)
    memory.consolidate_value = concise_consolidate
    optimizer = TextGradOptimizer(quiet=True)

    # Snapshot params once — all 8 problems share the same empty memory state
    train_params = recall_params(memory)

    # Fire all 8 LLM calls in parallel
    t_batch = time.time()
    with console.status(f"  Running {len(TRAIN_PROBLEMS)} training problems in parallel..."):
        batch_results = run_batch_parallel(TRAIN_PROBLEMS, train_params, generate_code)
    console.print(f"  Batch completed in {time.time() - t_batch:.1f}s\n")

    # Report results and run backward passes (sequential — optimizer is not thread-safe)
    result_nodes = []
    for i, (problem, (solution, exec_result, elapsed, result_node)) in enumerate(
        zip(TRAIN_PROBLEMS, batch_results)
    ):
        status = "[green]PASS[/]" if exec_result.passed else "[red]FAIL[/]"
        console.print(f"  [{i+1}/{len(TRAIN_PROBLEMS)}] {problem['id']}: {status}  ({elapsed:.1f}s)")
        feedback = build_feedback(problem, solution, exec_result)
        optimizer.backward(result_node, feedback)
        result_nodes.append(result_node)

    # Consolidate all gradients into memory in one step
    with console.status("  Consolidating gradients into memory..."):
        optimizer.consolidate(result_nodes[0])

    memory.close()
    memory = JSONMemoryBackend(LearningMemory, "demo", path=memory_path, quiet=True)
    memory.consolidate_value = concise_consolidate
    trained_params = recall_params(memory)

    console.print("\n[bold]Learned memory after 8 training examples:[/]")
    pbn = {p.source.name: p for p in trained_params}
    console.print(Panel(
        f"[bold cyan]coding_patterns:[/]\n{pbn['coding_patterns'].value}\n\n"
        f"[bold cyan]common_pitfalls:[/]\n{pbn['common_pitfalls'].value}",
        title="Trained Memory",
        border_style="green",
    ))

    # =========================================================================
    # STEP 3: Test with trained memory
    # =========================================================================
    console.print(Rule("[bold cyan]Step 3 — Test with Trained Memory[/]"))

    trained_results = {}
    for problem in TEST_PROBLEMS:
        with console.status(f"  Testing {problem['id']} (with memory)..."):
            solution, exec_result, elapsed, _ = run_problem(problem, trained_params, generate_code)

        status = "[green]PASS[/]" if exec_result.passed else "[red]FAIL[/]"
        console.print(f"  {problem['id']}: {status}  ({elapsed:.1f}s)")
        show_code(f"{problem['id']} — generated code (trained)", solution,
                  border="green" if exec_result.passed else "red")

        if not exec_result.passed:
            error_detail = exec_result.error or ""
            if exec_result.expected_output:
                error_detail += f"\n  Expected: {exec_result.expected_output}"
            if exec_result.actual_output:
                error_detail += f"\n  Got:      {exec_result.actual_output}"
            with console.status("  Explaining error..."):
                explanation = explain_error(
                    problem_prompt=problem["prompt"],
                    solution_code=solution,
                    error_message=error_detail,
                )
            console.print(Panel(
                f"[bold red]Error:[/] {error_detail}\n\n{explanation}",
                title="Why it failed", border_style="yellow",
            ))

        trained_results[problem["id"]] = {
            "solution": solution,
            "passed": exec_result.passed,
            "error": exec_result.error,
        }

    memory.close()

    # =========================================================================
    # Summary comparison
    # =========================================================================
    console.print(Rule("[bold]Summary[/]"))

    table = Table(title="Direct Test vs. Trained Test", border_style="bold")
    table.add_column("Problem", style="bold")
    table.add_column("Step 1: Direct", justify="center")
    table.add_column("Step 3: Trained", justify="center")
    table.add_column("Change", justify="center")

    for problem in TEST_PROBLEMS:
        pid = problem["id"]
        d = direct_results[pid]
        t = trained_results[pid]
        d_str = "[green]PASS[/]" if d["passed"] else "[red]FAIL[/]"
        t_str = "[green]PASS[/]" if t["passed"] else "[red]FAIL[/]"
        if not d["passed"] and t["passed"]:
            change = "[bold green]IMPROVED ✓[/]"
        elif d["passed"] and not t["passed"]:
            change = "[bold red]REGRESSED[/]"
        else:
            change = "—"
        table.add_row(pid, d_str, t_str, change)

    console.print(table)

    # Show before/after code diff for every improvement
    for problem in TEST_PROBLEMS:
        pid = problem["id"]
        d = direct_results[pid]
        t = trained_results[pid]
        if not d["passed"] and t["passed"]:
            console.print(f"\n[bold green]{pid} — FAIL → PASS[/]")
            show_code(f"{pid} before (FAIL)", d["solution"], border="red")
            show_code(f"{pid} after  (PASS)", t["solution"], border="green")

    Path(memory_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
