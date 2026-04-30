# Contributing to TT Spin Tracker

Thank you for your interest in contributing! This guide covers the development setup and contribution workflow.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/tt-spin-tracker.git
cd tt-spin-tracker
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install  # install git hooks
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html
```

## Code Style

This project uses `black` (formatting), `isort` (imports), and `flake8` (linting).
They run automatically on commit via pre-commit hooks.

To run manually:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Pull Request Process

1. Fork and create a feature branch: `git checkout -b feature/your-feature`
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/`
4. Update documentation if needed
5. Submit a PR with a clear description of changes

## Areas for Contribution

- 🎯 Better spin labeling tools (semi-automated annotation)
- 🏓 Multi-ball tracking (doubles rallies)
- 📱 Mobile-optimized inference (CoreML / TFLite export)
- 🎥 Support for additional camera angles
- 📊 Advanced analytics (serve pattern clustering, opponent modeling)
