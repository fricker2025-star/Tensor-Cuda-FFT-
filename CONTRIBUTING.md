# Contributing to FFT-Tensor

Thank you for your interest in contributing to FFT-Tensor! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/fft-tensor.git
cd fft-tensor
```

2. **Create a development environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e .[dev]
```

4. **Install pre-commit hooks:**
```bash
pre-commit install
```

## Code Style

- Follow PEP 8 for Python code
- Use Black for formatting: `black fft-tensor/`
- Use flake8 for linting: `flake8 fft-tensor/ --max-line-length=100`
- Use type hints where possible
- Document all public APIs with docstrings

## Testing

- Write tests for all new features
- Ensure all tests pass: `pytest tests/ -v`
- Aim for >90% code coverage
- Include both unit and integration tests

## CUDA Development

- Test on multiple GPU architectures if possible
- Use CUDA error checking macros (`CUDA_CHECK`, `CUFFT_CHECK`)
- Document kernel launch parameters and shared memory usage
- Profile performance with `nsys` or `nvprof`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear, atomic commits
3. Add tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass and code is formatted
6. Push to your fork and create a pull request

## Areas for Contribution

### High Priority
- **Performance**: Optimize CUDA kernels, add tensor core support
- **Features**: Adaptive sparsity, better spectral operators
- **Integration**: HuggingFace Transformers, JAX support

### Documentation
- Tutorials and examples
- API reference improvements
- Benchmark comparisons

### Research
- Novel sparsification strategies
- Frequency-domain training algorithms
- Applications to specific domains

## Reporting Issues

- Use the GitHub issue tracker
- Include minimal reproducible example
- Specify GPU model and CUDA version
- Include error messages and stack traces

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers feel welcome

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open a discussion on GitHub or reach out to the maintainers.

Thank you for contributing!
