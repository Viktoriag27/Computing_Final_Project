Contributing to DSDM_CaliHousePredict
We’re excited that you want to contribute to our project! Below, you'll find guidelines for contributing to ensure smooth collaboration.

Getting Started
1. Fork the Repository
Fork the repository to your own GitHub account by clicking the "Fork" button at the top right.
2. Clone the Fork
Clone your fork locally:
bash
Copy code
git clone https://github.com/Viktoriag27/Computing_Final_Project
cd DSDM_CaliHousePredict
3. Set Up the Project
Ensure you have Python 3.8+ installed.
Create and activate a virtual environment:
bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install the dependencies:
bash
Copy code
pip install -r requirements.txt
4. Install the Package Locally
Install the project in editable mode for development:
bash
Copy code
pip install -e .
5. Verify Installation
Run the test suite to ensure everything is set up correctly:
bash
Copy code
pytest tests/

Making Contributions

1. Create a Branch
Create a new branch for your contribution:
bash
Copy code
git checkout -b feature/your-feature-name
Use a descriptive branch name, such as feature/add-model or bugfix/fix-typo.
2. Follow Code Standards
Ensure your code follows the project conventions:
Use black for consistent formatting:
bash
Copy code
black .
Run flake8 to check for linting issues:
bash
Copy code
flake8 .
3. Write Tests
Add or update tests in the tests/ directory for any new features or changes.
Ensure all tests pass before submitting:
bash
Copy code
pytest --cov=DSDM_CaliHousePredict
4. Commit Changes
Write meaningful commit messages:
bash
Copy code
git commit -m "Add feature to predict house prices using CatBoost"
5. Push Changes
Push your branch to your fork:
bash
Copy code
git push origin feature/your-feature-name
6. Create a Pull Request
Go to the original repository and click "New Pull Request."
Provide a clear description of the changes you’ve made.
Reference any related issues (e.g., Closes #10).

Guidelines for Contributions

Bug Reports

Use the Issues section to report bugs.

Include steps to reproduce the bug and any relevant logs or screenshots.

Feature Requests

Open an issue for feature requests and describe:
What problem the feature solves.
How it would work.
Alternatives you’ve considered.

Code of Conduct

Follow our Code of Conduct to ensure a respectful and inclusive environment.
Need Help?
If you need assistance, feel free to:

Open an issue in the repository.
Email us alejandro.delgado@bse.eu
We appreciate your contributions and look forward to collaborating with you!