cat > .pre-commit-config.yaml << 'EOF'
# Pre-commit configuration for code quality automation
repos:
  # Black code formatter
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=100]

  # isort import sorter
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  # flake8 linter
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203,W503]

  # mypy type checker
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-r, survivex/, -f, json, -o, bandit-report.json]
        pass_filenames: false

  # Docstring formatting
  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
        args: [--convention=numpy]
        exclude: ^(tests/|setup.py)

# Configuration for specific hooks
ci:
  autofix_commit_msg: |
    [pre-commit.ci] auto fixes from pre-commit.com hooks

    for more information, see https://pre-commit.ci
  autofix_prs: true
  autoupdate_branch: ''
  autoupdate_commit_msg: '[pre-commit.ci] pre-commit autoupdate'
  autoupdate_schedule: weekly
  skip: []
  submodules: false
EOF
