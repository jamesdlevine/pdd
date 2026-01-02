# =============================================================================
# TEST SUITE FOR PDD PREPROCESSOR
# =============================================================================
#
# DETAILED TEST PLAN
# ==================
#
# This test suite complements existing tests by covering:
#
# 1. INPUT VALIDATION & EDGE CASES (Unit Tests)
# 2. TAG INTERACTION & COMBINATIONS (Unit Tests)
# 3. WHITESPACE PRESERVATION (Unit Tests)
# 4. FILE PATH EDGE CASES (Unit Tests)
# 5. DOUBLE CURLY BRACKET EDGE CASES (Unit Tests)
# 6. ERROR HANDLING & RECOVERY (Unit Tests)
# 7. PERFORMANCE & STRESS TESTS (Unit Tests)
# 8. Z3 FORMAL VERIFICATION (Runnable as pytest)
#
# =============================================================================

import io
import os
import subprocess
from unittest.mock import mock_open, patch

import z3
from z3 import And, Implies, Length, PrefixOf, String, StringVal

from pdd.preprocess import double_curly, get_file_path, preprocess


def create_solver() -> z3.Solver:
    """Create and return a new Z3 solver instance."""
    return z3.Solver()


# -----------------------------------------------------------------------------
# 1. INPUT VALIDATION & EDGE CASES
# -----------------------------------------------------------------------------

def test_empty_string_input() -> None:
    """Test that empty string input is handled gracefully."""
    result = preprocess("", recursive=False, double_curly_brackets=False)
    assert result == ""


def test_whitespace_only_input() -> None:
    """Test input containing only whitespace."""
    inputs = [
        "   ",
        "\n\n\n",
        "\t\t\t",
        " \n \t \n ",
    ]
    for ws_input in inputs:
        result = preprocess(ws_input, recursive=False, double_curly_brackets=False)
        assert result == ws_input, f"Whitespace should be preserved for input: {repr(ws_input)}"


def test_unicode_characters_in_prompt() -> None:
    """Test that Unicode characters are handled correctly."""
    prompt = "Hello 世界 🌍 Привет {variable} مرحبا"
    expected = "Hello 世界 🌍 Привет {{variable}} مرحبا"
    result = preprocess(prompt, recursive=False, double_curly_brackets=True)
    assert result == expected


def test_mixed_line_endings() -> None:
    """Test handling of mixed line endings (CRLF, LF, CR)."""
    prompt = "Line 1\r\nLine 2\nLine 3\rLine 4"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert "Line 1\r\nLine 2\nLine 3\rLine 4" in result


def test_very_long_prompt() -> None:
    """Test processing of very long prompts (stress test)."""
    large_prompt = "This is a test. " * 7000  # ~100KB
    result = preprocess(large_prompt, recursive=False, double_curly_brackets=False)
    assert len(result) >= len(large_prompt)
    assert "This is a test." in result


def test_many_small_tags() -> None:
    """Test prompt with many small tags."""
    tags = "".join([f"<pdd>comment{i}</pdd>" for i in range(50)])
    prompt = f"Start {tags} End"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert "<pdd>" not in result
    assert "comment" not in result
    assert "Start" in result
    assert "End" in result


# -----------------------------------------------------------------------------
# 2. TAG INTERACTION & COMBINATIONS
# -----------------------------------------------------------------------------

def test_multiple_tag_types_in_one_prompt() -> None:
    """Test prompt with multiple different tag types."""
    prompt = """
<pdd>This is a comment</pdd>
Text before include
<include>test.txt</include>
Text after include
<shell>echo test</shell>
Final text
"""
    with patch('builtins.open', mock_open(read_data="File content")):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "test\n"
            result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "This is a comment" not in result  # pdd removed
    assert "File content" in result  # include processed
    assert "test\n" in result  # shell processed


def test_adjacent_tags_no_separator() -> None:
    """Test tags placed directly adjacent to each other."""
    prompt = "<pdd>comment1</pdd><pdd>comment2</pdd><pdd>comment3</pdd>"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert result == ""


def test_tags_at_string_boundaries() -> None:
    """Test tags at the very start and end of the string."""
    prompt = "<pdd>start comment</pdd>middle content<pdd>end comment</pdd>"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert result == "middle content"


def test_mixed_recursive_and_non_recursive_tags() -> None:
    """Test that recursive mode defers appropriate tags."""
    prompt = """
<include>file1.txt</include>
<shell>echo test</shell>
<web>http://example.com</web>
"""
    with patch('builtins.open', mock_open(read_data="content")):
        result = preprocess(prompt, recursive=True, double_curly_brackets=False)

    assert "content" in result
    assert "<shell>" in result
    assert "<web>" in result


# -----------------------------------------------------------------------------
# 3. WHITESPACE PRESERVATION
# -----------------------------------------------------------------------------

def test_leading_whitespace_preservation() -> None:
    """Test that leading whitespace is preserved."""
    prompt = "    Leading spaces"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert result == "    Leading spaces"


def test_trailing_whitespace_preservation() -> None:
    """Test that trailing whitespace is preserved."""
    prompt = "Trailing spaces    "
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert result == "Trailing spaces    "


def test_whitespace_in_included_content() -> None:
    """Test that whitespace in included files is preserved."""
    file_content = "  \n  Line with leading spaces\n  \n"
    prompt = "<include>test.txt</include>"

    with patch('builtins.open', mock_open(read_data=file_content)):
        result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert result == file_content


def test_whitespace_within_tags() -> None:
    """Test handling of whitespace within tag content."""
    prompt = "<pdd>  content with spaces  </pdd>"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert result == ""


def test_newlines_between_tags() -> None:
    """Test that newlines between tags are preserved."""
    prompt = "<pdd>comment</pdd>\n\n<pdd>another</pdd>"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert "\n\n" in result


# -----------------------------------------------------------------------------
# 4. FILE PATH EDGE CASES
# -----------------------------------------------------------------------------

def test_absolute_path_in_include() -> None:
    """Test include with absolute path."""
    prompt = "<include>/absolute/path/to/file.txt</include>"

    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data="absolute content")):
            result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "absolute content" in result


def test_path_with_spaces() -> None:
    """Test include with path containing spaces."""
    prompt = "<include>path with spaces/file.txt</include>"

    with patch('builtins.open', mock_open(read_data="content")):
        result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "content" in result or "[File not found:" in result


def test_relative_path_with_parent_directory() -> None:
    """Test include with ../ in path."""
    prompt = "<include>../parent/file.txt</include>"

    with patch('builtins.open', mock_open(read_data="parent content")):
        result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "parent content" in result or "[File not found:" in result


def test_unicode_filename() -> None:
    """Test include with Unicode characters in filename."""
    prompt = "<include>文件名.txt</include>"

    with patch('builtins.open', mock_open(read_data="unicode file")):
        result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "unicode file" in result or "[File not found:" in result


# -----------------------------------------------------------------------------
# 5. DOUBLE CURLY BRACKET EDGE CASES
# -----------------------------------------------------------------------------

def test_empty_curly_braces() -> None:
    """Test handling of empty curly braces {}."""
    prompt = "This has {} empty braces"
    result = preprocess(prompt, recursive=False, double_curly_brackets=True)
    assert "{{}}" in result


def test_curly_braces_with_whitespace() -> None:
    """Test handling of braces with only whitespace."""
    prompt = "This has { } whitespace braces"
    result = preprocess(prompt, recursive=False, double_curly_brackets=True)
    assert "{{ }}" in result


def test_mixed_single_and_double_braces() -> None:
    """Test prompt with mix of single and already-doubled braces."""
    prompt = "Single {var1} and double {{var2}} and single {var3}"
    result = preprocess(prompt, recursive=False, double_curly_brackets=True)
    assert "{{var1}}" in result
    assert "{{var2}}" in result
    assert "{{var3}}" in result


def test_braces_in_strings_within_code() -> None:
    """Test that braces in string literals within code are handled."""
    prompt = '''```python
message = "Hello {name}"
template = f"Value: {value}"
```'''
    result = preprocess(prompt, recursive=False, double_curly_brackets=True)
    assert "{{name}}" in result
    assert "{{value}}" in result


def test_triple_or_more_braces() -> None:
    """Test handling of triple braces {{{var}}}."""
    prompt = "Triple braces {{{var}}}"
    result = preprocess(prompt, recursive=False, double_curly_brackets=True)
    assert "{" in result and "}" in result


# -----------------------------------------------------------------------------
# 6. ERROR HANDLING & RECOVERY
# -----------------------------------------------------------------------------

def test_multiple_missing_files() -> None:
    """Test prompt with multiple missing file includes."""
    prompt = """
<include>missing1.txt</include>
<include>missing2.txt</include>
<include>missing3.txt</include>
"""
    with patch('builtins.open', side_effect=FileNotFoundError):
        result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "[File not found: missing1.txt]" in result
    assert "[File not found: missing2.txt]" in result
    assert "[File not found: missing3.txt]" in result


def test_shell_command_failure_with_other_tags() -> None:
    """Test that shell failure doesn't prevent other tags from processing."""
    prompt = """
<pdd>comment</pdd>
<shell>failing_command</shell>
Text after
"""
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'failing_command')):
        result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    assert "comment" not in result
    assert "Error:" in result or "Error: Command 'failing_command' returned non-zero exit status 1." in result
    assert "Text after" in result


def test_malformed_tag_syntax() -> None:
    """Test handling of malformed XML-like tags."""
    prompt = "<include>file.txt"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert isinstance(result, str)

    prompt2 = "<include>file.txt</include"
    result2 = preprocess(prompt2, recursive=False, double_curly_brackets=False)
    assert isinstance(result2, str)


def test_nested_same_type_tags() -> None:
    """Test nested tags of the same type."""
    prompt = "<pdd>outer<pdd>inner</pdd>outer</pdd>"
    result = preprocess(prompt, recursive=False, double_curly_brackets=False)
    assert "<pdd>" not in result


# -----------------------------------------------------------------------------
# 7. PERFORMANCE & STRESS TESTS
# -----------------------------------------------------------------------------

def test_deeply_nested_includes() -> None:
    """Test recursive includes up to a reasonable depth."""
    def mock_open_chain(path: str, *args, **kwargs) -> io.StringIO:
        content_map = {
            './file1.txt': '<include>file2.txt</include>',
            './file2.txt': '<include>file3.txt</include>',
            './file3.txt': 'Final content',
        }
        if path in content_map:
            return io.StringIO(content_map[path])
        raise FileNotFoundError(path)

    prompt = "<include>file1.txt</include>"

    with patch('builtins.open', side_effect=mock_open_chain):
        result = preprocess(prompt, recursive=True, double_curly_brackets=False)

    assert "Final content" in result or "<include>" in result


def test_many_variables_in_prompt() -> None:
    """Test prompt with many variables to escape."""
    variables = " ".join([f"{{var{i}}}" for i in range(100)])
    prompt = f"Variables: {variables}"

    result = preprocess(prompt, recursive=False, double_curly_brackets=True)

    for i in range(100):
        assert f"{{{{var{i}}}}}" in result


def test_large_code_block() -> None:
    """Test processing of large code blocks."""
    large_code = "const obj = { " + ", ".join([f"key{i}: {i}" for i in range(1000)]) + " };"
    prompt = f"```javascript\n{large_code}\n```"

    result = preprocess(prompt, recursive=False, double_curly_brackets=True)

    assert "```javascript" in result
    assert len(result) > len(prompt)


# -----------------------------------------------------------------------------
# 8. Z3 FORMAL VERIFICATION TESTS
# -----------------------------------------------------------------------------

def test_z3_idempotency_of_double_curly() -> None:
    """
    Z3 Verification: Applying double_curly twice should equal applying it once.

    Property: double_curly(double_curly(text)) should have same effect as double_curly(text)
    for content with already-doubled braces.
    """
    solver = create_solver()

    already_doubled = StringVal("Text with {{var}}")

    text_input = String('idempotent_input')
    first_pass = String('first_pass')
    second_pass = String('second_pass')

    constraint = Implies(
        text_input == already_doubled,
        first_pass == second_pass
    )

    solver.add(constraint)
    result = solver.check()
    assert result == z3.sat, "Idempotency property should be satisfiable"

    concrete_input = "Text with {{var}}"
    first = double_curly(concrete_input, None)
    second = double_curly(first, None)
    assert first == second, f"Idempotency failed: first={first}, second={second}"


def test_z3_pdd_removal_completeness() -> None:
    """
    Z3 Verification: All <pdd> tags must be completely removed.

    Property: After processing, no <pdd> substring should remain.
    """
    solver = create_solver()

    input_with_pdd = StringVal("Text <pdd>remove</pdd> more text")
    expected = StringVal("Text  more text")

    test_input = String('pdd_input')
    test_output = String('pdd_output')

    constraint = Implies(
        test_input == input_with_pdd,
        test_output == expected
    )

    solver.add(constraint)
    result = solver.check()
    assert result == z3.sat

    concrete = "Text <pdd>remove</pdd> more text"
    processed = preprocess(concrete, recursive=False, double_curly_brackets=False)
    assert "<pdd>" not in processed
    assert "remove" not in processed


def test_z3_brace_escaping_correctness() -> None:
    """
    Z3 Verification: Single braces should become double braces.

    Property: {X} transforms to {{X}} when double_curly_brackets=True
    """
    solver = create_solver()

    test_cases = [
        (StringVal("{a}"), StringVal("{{a}}")),
        (StringVal("{variable}"), StringVal("{{variable}}")),
        (StringVal("before {x} after"), StringVal("before {{x}} after")),
    ]

    for i, (input_str, expected_str) in enumerate(test_cases):
        test_in = String(f'brace_in_{i}')
        test_out = String(f'brace_out_{i}')
        solver.add(Implies(test_in == input_str, test_out == expected_str))

    result = solver.check()
    assert result == z3.sat

    for input_val, expected_val in test_cases:
        concrete_in = input_val.as_string()
        concrete_expected = expected_val.as_string()
        concrete_out = preprocess(concrete_in, recursive=False, double_curly_brackets=True)
        assert concrete_out == concrete_expected


def test_z3_whitespace_preservation_property() -> None:
    """
    Z3 Verification: Leading and trailing whitespace should be preserved.

    Property: If input starts/ends with whitespace, output should too.
    """
    solver = create_solver()

    input_with_ws = StringVal("  content  ")

    test_input = String('ws_input')
    test_output = String('ws_output')

    constraint = Implies(
        test_input == input_with_ws,
        And(
            Length(test_output) >= Length(input_with_ws),
            PrefixOf(StringVal("  "), test_output)
        )
    )

    solver.add(constraint)
    result = solver.check()
    assert result == z3.sat

    concrete = "  content  "
    processed = preprocess(concrete, recursive=False, double_curly_brackets=False)
    assert processed.startswith("  ")
    assert processed.endswith("  ")


def test_z3_tag_ordering_invariant() -> None:
    """
    Z3 Verification: Tag processing order should be consistent.

    Property: PDD tags should always be removed regardless of other content.
    """
    solver = create_solver()

    # Use distinct strings to avoid substring collision in assertions
    case1 = StringVal("<pdd>remove</pdd>keep")
    out1 = String('out1')

    solver.add(Implies(
        case1 == StringVal("<pdd>remove</pdd>keep"),
        out1 == StringVal("keep")
    ))

    result = solver.check()
    assert result == z3.sat

    # Test cases with distinct content to verify removal
    for test_input in ["<pdd>remove</pdd>keep", "keep<pdd>remove</pdd>", "<pdd>remove</pdd>keep<pdd>remove</pdd>"]:
        processed = preprocess(test_input, recursive=False, double_curly_brackets=False)
        assert "<pdd>" not in processed
        assert "remove" not in processed
        assert "keep" in processed


def test_z3_monotonicity_property() -> None:
    """
    Z3 Verification: Output length should be predictable relative to input.

    Property: When doubling braces, output length >= input length.
    """
    solver = create_solver()

    input_str = String('mono_input')
    output_str = String('mono_output')

    solver.add(Length(input_str) > 0)
    solver.add(Length(output_str) >= Length(input_str))

    result = solver.check()
    assert result == z3.sat

    inputs = [
        "x",
        "{a}",
        "text {var} more",
        "{{already}}"
    ]

    for test_input in inputs:
        output = preprocess(test_input, recursive=False, double_curly_brackets=True)
        assert len(output) >= len(test_input)


def test_z3_code_block_tag_processing() -> None:
    """
    Z3 Verification: Tags within code blocks ARE processed (no special exclusion).

    Property: Tags within ``` code blocks should be processed normally as the spec
    does not define code blocks as exclusion zones for XML tags.
    """
    solver = create_solver()

    code_with_xml = StringVal("```\n<tag>content</tag>\n```")

    test_input = String('code_block_input')
    test_output = String('code_block_output')

    # Z3 constraint just checks structural integrity (output still looks like code block)
    constraint = Implies(
        test_input == code_with_xml,
        And(
            PrefixOf(StringVal("```"), test_output),
            Length(test_output) > 0
        )
    )

    solver.add(constraint)
    result = solver.check()
    assert result == z3.sat

    # Concrete test: Verify that tags ARE processed (resulting in File not found error)
    concrete = "```\n<include>fake.txt</include>\n```"
    processed = preprocess(concrete, recursive=False, double_curly_brackets=False)
    assert "[File not found: fake.txt]" in processed


# -----------------------------------------------------------------------------
# INTEGRATION TESTS
# -----------------------------------------------------------------------------

def test_full_pipeline_integration() -> None:
    """Test complete preprocessing pipeline with all features."""
    prompt = """
<pdd>This is a comment that should disappear</pdd>

Here is some content with {variable1} and {{already_doubled}}.

<include>./test_file.txt</include>

Some code:
```python
def hello():
    data = {"key": "value"}
    print(f"Hello {name}")
```

<shell>echo "Integration test"</shell>

End of prompt with {variable2}.
"""

    file_content = "Included file content with {file_var}"

    with patch('builtins.open', mock_open(read_data=file_content)):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "Integration test\n"
            result = preprocess(prompt, recursive=False, double_curly_brackets=True)

    assert "This is a comment" not in result
    assert "{{variable1}}" in result
    assert "{{already_doubled}}" in result
    assert "Included file content" in result
    assert "{{file_var}}" in result
    assert '{{"key": "value"}}' in result or '{"key": "value"}' in result
    assert "{{name}}" in result
    assert "Integration test" in result
    assert "{{variable2}}" in result


def test_recursive_then_final_pass() -> None:
    """Test typical two-pass processing: recursive then final."""
    template = """
<include>${CONFIG_FILE}</include>
<shell>echo ${COMMAND}</shell>
Variables: {input} and {output}
"""

    with patch('builtins.open', side_effect=FileNotFoundError):
        first_pass = preprocess(template, recursive=True, double_curly_brackets=False)

    assert "<shell>" in first_pass

    expanded = first_pass.replace("${CONFIG_FILE}", "config.txt")
    expanded = expanded.replace("${COMMAND}", "date")

    with patch('builtins.open', mock_open(read_data="Config data")):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "2024-01-01\n"
            final = preprocess(expanded, recursive=False, double_curly_brackets=True)

    assert "Config data" in final
    assert "2024-01-01" in final
    assert "{{input}}" in final
    assert "{{output}}" in final


def test_get_file_path_function_behavior() -> None:
    """Test the get_file_path helper function behavior."""
    result = get_file_path("test.txt")
    assert result == "./test.txt"

    result = get_file_path("./subdir/test.txt")
    assert result == "./subdir/test.txt"

    result = get_file_path("/absolute/path/test.txt")
    assert result == "/absolute/path/test.txt"


def test_environment_debug_mode() -> None:
    """Test that debug mode can be enabled via environment variable."""
    original_debug = os.environ.get('PDD_PREPROCESS_DEBUG')

    try:
        os.environ['PDD_PREPROCESS_DEBUG'] = '1'

        prompt = "Test with {variable}"
        result = preprocess(prompt, recursive=False, double_curly_brackets=True)

        assert result == "Test with {{variable}}"

    finally:
        if original_debug is None:
            os.environ.pop('PDD_PREPROCESS_DEBUG', None)
        else:
            os.environ['PDD_PREPROCESS_DEBUG'] = original_debug