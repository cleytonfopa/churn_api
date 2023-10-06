# Churn API

# Welcome to Churn API!

This API enables real-time generation of the probability of *churn* for
a specific user. By using the user’s activity data within the betting
platform during its initial days, a machine learning model forecasts the
expected probability of churning.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Endpoints](#endpoints)
  - [Endpoint 1](#endpoint-1)
- [Authors](#authors)

## Getting Started <a name="getting-started"></a>

### Prerequisites <a name="prerequisites"></a>

This API is written in Python. In order to set up all dependencies, it
is essential you to use the Python virtualenv management tool
[`pipenv`](https://pipenv.pypa.io/en/latest/).

### Installation <a name="installation"></a>

Once you clone this repository, go the API directory. There will be a
file called `Pipfile.lock` specifying all the dependencies used. Open a
terminal and run the following command:

    pipenv sync

This command will install all the packages and their respective versions
necessary to run the API.

### Usage <a name="usage"></a>

Once the environment is set up, we can run the API as follows:

    pipenv run python api.py

## Endpoints <a name="endpoints"></a>

### Endpoint 1 <a name="endpoint-1"></a>

- **URL**: `/predict_churn`

- **Method**: POST

- **Description**: This endpoint will calculate the prediction for the
  probability of churning for each player.

- **Input**: `JSON` file containing the **merge between two datasets:
  *casino* and *players***. See the template file `test.json` in the
  `examples/` folder.

- **Response**:

  - JSON with the point estimate for the probability of churn, as well
    as the lower and upper bound prediction the for each player.

  For example, for two players:
  `[{"Username":"erika12144","churn_proba": 0.873},{"Username":"euphobravo","churn_proba": 0.311}]`

## Examples <a name="examples"></a>

There is a file called `test.json` in the `examples/` folder in this
repository that contains an example of the input format this API accept.

Suppose you are running the API locally on default port `1234`. You can
make a request to the API as follows:

    curl -XPOST -H "Content-Type: application/json" -d @examples/test.json http://127.0.0.1:5000/predict_churn

## Author <a name="authors"></a>

- [Cleyton Farias](mailto:cleytonfarias@outlook.com "e-mail");
