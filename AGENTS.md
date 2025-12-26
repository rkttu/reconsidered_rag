# AGENTS.md

This file contains essential information for AI agents working on this project.

## Package Manager

This project uses **uv** as its package manager. Do **NOT** use `pip` directly.

### Installing Packages

```bash
# Add a package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Sync dependencies (install from lock file)
uv sync

# Remove a package
uv remove <package-name>
```

### Important Notes

- Always use `uv add` instead of `pip install`.
- Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.
- The virtual environment is located in the `.venv` directory.
- Do not modify `uv.lock` manually; it is auto-generated.

## Running Python

### Executing Scripts

```bash
# Run a Python script
uv run python <script.py>

# Run a module
uv run python -m <module_name>
```

### Activating Virtual Environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate
```

## Project Structure

- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions (do not edit manually)
- `.venv/` - Virtual environment directory
- `.python-version` - Python version specification (if present)

## Common Commands

| Task | Command |
| ------ | --------- |
| Install all dependencies | `uv sync` |
| Add a package | `uv add <pkg>` |
| Add dev dependency | `uv add --dev <pkg>` |
| Remove a package | `uv remove <pkg>` |
| Run a script | `uv run python <script.py>` |
| Update all packages | `uv lock --upgrade` then `uv sync` |
| Show installed packages | `uv pip list` |

## Best Practices

1. **Always sync after cloning**: Run `uv sync` after cloning the repository.
2. **Commit lock file**: Always commit `uv.lock` to version control.
3. **Use `uv run`**: Prefer `uv run python` to ensure correct environment.
4. **Check pyproject.toml**: Verify dependencies are added to `pyproject.toml` after `uv add`.

## Code Quality

This project uses **Pylance** (VS Code Python language server) for code validation.

### Required Workflow

After writing or modifying Python code, you **MUST**:

1. **Check for errors**: Review any syntax errors or warnings reported by Pylance.
2. **Resolve all issues**: Fix any reported problems before considering the task complete.
3. **Verify imports**: Ensure all imports are valid and packages are installed.

### Common Pylance Checks

- Type errors and type mismatches
- Undefined variables or imports
- Unused imports and variables
- Missing function arguments
- Invalid syntax

### Validation Steps

- Use the `get_errors` tool to retrieve current diagnostics.
- Review the Problems panel in VS Code.
- Ensure zero errors before finalizing changes.

## Markdown Guidelines

When writing or editing Markdown files, follow **markdownlint** rules:

### Key Rules

- Use consistent heading levels (no skipping levels)
- Add blank lines around headings, code blocks, and lists
- Use consistent list markers (`-` or `*`)
- No trailing spaces at end of lines
- Single blank line at end of file
- No multiple consecutive blank lines
- Proper indentation for nested lists (2 or 4 spaces)

### Linting Steps

- Review any markdownlint warnings in VS Code.
- Fix all linting issues before finalizing changes.
