# Contributing to Customer Churn Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Writing Code

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions and modules
- Keep functions focused and testable

### Code Quality

```bash
# Format code with Black
black churn_prediction.py

# Lint with flake8
flake8 churn_prediction.py --max-line-length=100

# Type checking with mypy
mypy churn_prediction.py
```

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=.

# Generate HTML report
pytest --cov --cov-report=html
```

### Documentation

- Update README.md for major changes
- Add docstrings to all functions
- Include usage examples for new features
- Update this file if changing contribution process

## Commit Guidelines

Write clear commit messages:

```
[type] Brief description (50 chars max)

Detailed explanation of changes (72 chars per line)

Closes #issue_number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Build, dependencies, etc

Example:
```
feat: Add hyperparameter tuning with GridSearchCV

Implement GridSearchCV for hyperparameter optimization across
all three models. Includes cross-validation and best model selection.

Closes #42
```

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Push your changes**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Clear title and description
   - Link to related issues
   - Reference any breaking changes
   - Add screenshots/examples if visual changes

4. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Related Issues
   Closes #issue_number

   ## Changes
   - Bullet point 1
   - Bullet point 2

   ## Testing
   How to test these changes

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] No new warnings generated
   ```

## Areas for Contribution

### High Priority
- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Implement feature selection techniques
- [ ] Add unit tests and pytest integration
- [ ] Create CI/CD pipeline (GitHub Actions)
- [ ] Add data drift monitoring

### Medium Priority
- [ ] Add explainability features (SHAP values)
- [ ] Implement cross-validation strategy comparison
- [ ] Create API wrapper for model serving
- [ ] Add interactive dashboard (Plotly/Streamlit)
- [ ] Performance optimization

### Low Priority
- [ ] Documentation improvements
- [ ] Example notebooks
- [ ] Additional visualizations
- [ ] Code refactoring

## Review Process

1. Maintainers will review your PR within 7 days
2. Address feedback and push updates
3. Once approved, your PR will be merged
4. Your contribution will be credited

## Reporting Issues

Use GitHub Issues with:
- Clear title
- Detailed description
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment info (Python version, OS, etc.)

### Issue Template
```markdown
## Description
Brief description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- OS:
- Package versions: (pip show package-name)

## Screenshots
If applicable
```

## Documentation

### Docstring Format (Google Style)

```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Brief one-line description.
    
    Longer description explaining the function's purpose,
    behavior, and any important details.
    
    Parameters:
    -----------
    arg1 : str
        Description of arg1
    arg2 : int
        Description of arg2
    
    Returns:
    --------
    bool
        Description of return value
    
    Raises:
    -------
    ValueError
        When arg2 is negative
    
    Examples:
    ---------
    >>> result = function_name("test", 5)
    >>> print(result)
    True
    """
```

## Questions?

- Open a Discussion on GitHub
- Check existing issues and PRs
- Read the documentation in docs/

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing! 🎉
