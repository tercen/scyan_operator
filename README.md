
# Template Python Operator

Cell-wise mean calculated implemented in Python.

## Python operator - Development workflow

* Set up [the Tercen Studio development environment](https://github.com/tercen/tercen_studio)
* Create a new git repository based on the [template Python operator](https://github.com/tercen/template-python-operator)
* Open VS Code Server by going to: http://127.0.0.1:8443
* Clone this repository into VS Code (using the 'Clone from GitHub' command from the Command Palette for example)
* Load the environment and install core requirements by running the following commands in the terminal:

```bash
source /config/.pyenv/versions/3.9.0/bin/activate
pip install -r requirements.txt
```

* Develop your operator. Note that you can interact with an existing data step by specifying arguments to the `TercenContext` function:

```python
tercenCtx = ctx.TercenContext()
```

```python
tercenCtx = ctx.TercenContext(
    workflowId="YOUR_WORKFLOW_ID",
    stepId="YOUR_STEP_ID",
    username="admin", # if using the local Tercen instance
    password="admin", # if using the local Tercen instance
    serviceUri = "http://127.0.0.1:5400/" # if using the local Tercen instance 
)
```

* Generate requirements

```bash
python3 -m tercen.util.requirements . > requirements.txt
```

* Push your changes to GitHub: triggers CI GH workflow
* Tag the repository: triggers Release GH workflow
* Go to tercen and install your operator


## Helpful Commands

### Install Tercen Python Client

```bash
python3 -m pip install --force git+https://github.com/tercen/tercen_python_client@0.7.1
```

### Wheel

Though not strictly mandatory, many packages require it.

```bash
python3 -m pip install wheel
```

### VSCode Launch

To run and debug the code, `VSCode` requires a launch.json file, which will be automatically created.
If the generated file does not run properly within the virtual environment, use the one below:

```JSON
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "python": "PATH TO VENV PYTHON",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": { "PYTHONPATH": "${workspaceRoot}"}
            
        }
    ]
}
```

    