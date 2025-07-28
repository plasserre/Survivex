cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "survivex"
version = "0.1.0"
description = "Advanced survival analysis library for Python with GPU acceleration and comprehensive statistical methods"
authors = [
    {name = "Surviv Contributors", email = "maintainers@survivex.org"}
]
maintainers = [
    {name = "Surviv Core Team", email = "core@survivex.org"}
]
license = {text = "Apache-2.0"}
readme = "README.md"
requires-python = ">=3.8"
keywords = [
    "survival analysis",
    "statistics", 
    "biostatistics",
    "machine learning",
    "gpu acceleration",
    "pytorch",
    "kaplan-meier",
    "cox regression",
    "multi-state models"
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Healthcare Industry", 
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "tqdm>=4.60.0",
    "scikit-learn>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "nbsphinx>=0.9.0",
    "jupyter>=1.0.0",
]
gpu = [
    "cupy>=12.0.0",
]
viz = [
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    "ipywidgets>=8.0.0",
]
benchmark = [
    "lifelines>=0.27.0",
    "scikit-survival>=0.20.0",
    "memory-profiler>=0.60.0",
]
all = ["surviv[dev,gpu,viz,benchmark]"]

[project.urls]
Homepage = "https://github.com/survivex-py/survivex"
Documentation = "https://survivex.readthedocs.io"
Repository = "https://github.com/survivex-py/survivex"
"Bug Tracker" = "https://github.com/survivex-py/surviv/issues"
"Discussions" = "https://github.com/survivex-py/surviv/discussions"
Changelog = "https://github.com/survivex-py/surviv/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["."]
include = ["surviv*"]
exclude = ["tests*", "docs*", "examples*"]

[tool.setuptools.package-data]
surviv = ["datasets/data/*", "py.typed"]

[tool.black]
line-length = 100
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = [
    "-v",
    "--cov=survivex",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=90",
    "--strict-markers",
    "--strict-config",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as benchmarks",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["survivex"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
EOF
