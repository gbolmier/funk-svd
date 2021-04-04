# Contributor guide

**Table of contents:**
- [Installation](#installation)
  - [a) Create an isolated environment](#a-create-an-isolated-environment)
  - [b) Setup a sync copy of the project repository](#b-setup-a-sync-copy-of-the-project-repository)
  - [c) Install `funk-svd` in development mode](#c-install-funk-svd-in-development-mode)
- [Making changes](#making-changes)
  - [a) Create a feature branch](#a-create-a-feature-branch)
  - [b) Make changes](#b-make-changes)
  - [c) Development conventions and testing](#c-development-conventions-and-testing)
- [Submitting changes](#submitting-changes)

*If you aren't comfortable using [Git](https://git-scm.com/) or some of its functionalities, the [Pro Git book](https://git-scm.com/book/en) (within Git documentation) is a great resource where you can cherry-pick any subject you need to understand.*

## Installation

### a) Create an isolated environment

Optional, but highly recommended, it's good practice to keep required dependencies separated from other projects by creating an isolated environment.

Popular choices include [virtual environments](https://docs.python.org/3/library/venv.html), from the Python standard library:

```
$ python -m venv <env_path>
$ source <env_path/bin/activate>
```

and [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):

```
$ conda create --name <env_name> "python>=3.6.5,<3.9"
$ conda activate <env_name>
```

### b) Setup a sync copy of the project repository

It boils down to fork, clone, and sync:

- You first want to fork the project repository (i.e. creating a personal server-side copy): <br>
go to the [project repository webpage](https://github.com/gbolmier/funk-svd) and click the <kbd>Fork</kbd> button on the top right of the page.

- Then clone your fork to your local machine (i.e. creating a local copy on your machine): <br>
```$ git clone https://github.com/your-username/funk-svd.git```

- And finally [sync your fork](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) with the upstream repository (the "central" server-side repository, i.e. the parent of the fork): <br>
```$ cd funk-svd``` <br> ```$ git remote add upstream https://github.com/gbolmier/funk-svd.git```

The whole process is also well [documented by GitHub](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo).

### c) Install `funk-svd` in development mode

Navigate to the cloned directory and install the library in editable mode so that changes in the code take effect immediately, and with the required development dependencies (cf. the following [stackoverflow question](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install)):

```
$ pip install -e ".[dev]"
```


## Making changes

### a) Create a feature branch

Now that our fork is synced with upstream we can update our local master branch with upstream latest changes:

```
$ git checkout master
$ git pull upstream master
```

It's good practice to make changes within an independent line of development so that the master branch reflects only production-ready code. Let's create a new feature branch and tell git to point to this latter:

```
$ git branch <new_feature>
$ git checkout <new_feature>
```

### b) Make changes

Develop within your feature branch and [record your changes to the repository](https://git-scm.com/book/en/Git-Basics-Recording-Changes-to-the-Repository):

```
$ git add <modified_files>
$ git commit
```

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

```
$ git push -u origin <new_feature>
```

If the changes you're working on might be impacted by potential changes in the upstream repository, you probably want to merge upstream latest changes into your local feature branch before and after editing it — enabling you to detect conflicts or changes breaking yours, early. To do so:

```
$ git fetch upstream
$ git merge upstream/master
```

If you aren't familiar with conflict solving, you can refer to the [related Github documentation](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line).

### c) Development conventions and testing

The project follows [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) for Python code, and [NumPy format](https://numpydoc.readthedocs.io/en/latest/format.html) for docstring conventions. Unit testing is done with the [`pytest`](https://docs.pytest.org/en/stable/) framework.

Before being merged, changes must pass PEP8 and unit testing checks, both executed from the root of the project:
```
$ flake8
$ pytest
```

**Clarity** and **conciseness** are warmly encouraged. One asset of Python being its high **readability**, it would be too bad not taking advantage of it. We human beings have limited **cognitive load**, meaning that we can't remember too many items in our short term memory. In order to make the development experience more friendly for everyone, let's create **abstractions** by **grouping** concept-related instructions together and give them relevant names. This process of **chunking** and **aliasing** makes grouped items more easily remembered, reducing our **cognitive load**. Said in a more practical maneer, it consist of:

- **grouping** instructions by **concepts** in variables, functions, and classes
- giving **explicit** instead of cryptic names
- defining the **abstractions** right level of **granularity**, avoiding too long or too nested series of instructions

For example, instead of:

```python
grades = [3.25, 2, 4.5, 3.75, 5]
m = sum(grades) / len(grades)
std = (sum(((grade - m)**2 for grade in grades)) / len(grades))**.5
```

prefer something like:

```python
grades = [3.25, 2, 4.5, 3.75, 5]

def standard_deviation(x: Sequence[Real]) -> float:
    n = len(x)
    x_mean = sum(x) / n
    x_centered_squared = ((xi - x_mean)**2 for xi in x)
    variance = sum(x_centered_squared) / n
    return variance**.5

grades_std = standard_deviation(grades)
```

Obviously, code readability must be balanced with code performance depending on the purpose of the code. Therefore, when you are forced to write complex code (e.g. to make it faster), add clear comments of what it does under the hoods:

```python
def fast_standard_deviation(x: Sequence[Real]) -> float:
    x = np.array(x)
    # Center and square x vector
    # Compute the arithmetic mean of the previous result
    # Return the square root of the previous result
    return np.sqrt(((x - x.mean())**2).mean())
```


## Submitting changes

Follow Github documentation instructions to [create a pull request from your fork and submit it to the upstream repository](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). Prefix your pull request name by `[WIP]` if it's still work in progress, or `[MRG]` once you consider it ready to be merged. In the latter case, don't forget to synchronize your feature branch with the latest changes.
