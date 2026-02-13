# Insurance Charge Prediction API

This project trains a linear regression model to predict insurance charges and exposes a simple Flask API.

## Running the server

```powershell
python clf.py         # trains the model and saves insurance_model.pkl
python app.py         # starts the Flask server on http://127.0.0.1:5000
```

## Making predictions

The `/predict` endpoint expects a JSON body with an `input` key whose value is a list of exactly six elements in the following order:

1. `age` (numeric)
2. `bmi` (numeric)
3. `children` (numeric)
4. `sex` (`male` or `female`)
5. `smoker` (`yes` or `no`)
6. `region` (e.g. `southwest`)

> **Important:** the API converts the list into a pandas DataFrame with the appropriate column names before passing it through the pipeline. Sending just a raw array to the model (as a NumPy array) causes the `ColumnTransformer` to complain:
> ```
> Specifying the columns using strings is only supported for dataframes.
> ```

### Example POST request (using Postman or curl)

```http
POST /predict HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
  "input": [26, 27.9, 0, "female", "yes", "southwest"]
}
```

A successful response looks like:

```json
{"prediction": 18859}
```

If you omit or mis-order the fields, the server will return an error message indicating the problem.

