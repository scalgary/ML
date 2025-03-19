def summarize(results,
              conf_int=False,
              level=None):
    from io import StringIO
    import pandas as pd
   
    """
    Take a fit statsmodels and summarize it
    by returning the usual coefficient estimates,
    their standard errors, the usual test
    statistics and P-values as well as 
    (optionally) confidence intervals.

    Based on:

    https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe

    Parameters
    ----------
    results : a results object (
    
    conf_int : bool (optional)
        Include 95% confidence intervals?

    level : float (optional)
        Confidence level (default: 0.95)
   
    """
        
    if level is not None:
        conf_int = True
    if level is None:
        level = 0.95
    tab = results.summary(alpha=1-level).tables[1]
    results_table = pd.read_html(StringIO(tab.as_html()),
                                 index_col=0,
                                 header=0)[0]
    if not conf_int:
        columns = ['coef',
                   'std err',
                   't',
                   'P>|t|']
        return results_table[results_table.columns[:-2]]
    return results_table




def transform_df_for_model(df, terms, interactions=None, contrast ="drop"):
    import pandas as pd
    import numpy as np
    import formulaic
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    """
    Transforms a DataFrame into a design matrix for modeling using Formulaic,
    supporting both "drop" (one-hot encoding) and "sum" (contrast sum coding).

    Features:
    - Detects categorical variables and encodes them using "drop" or "sum" encoding.
    - Ensures the final column order matches `variables`, replacing categorical variables in the correct position.
    - Supports interaction terms while keeping order.
    - Renames categorical variables in the format `oldvariable[value]`.

    Parameters:
     df : pd.DataFrame
        The DataFrame containing the dataset.
    variables : list
        A list of column names to include in the design matrix (order is preserved).
    interactions : list of tuples, optional
        A list of tuples specifying interaction terms (e.g., [('lstat', 'rm')] for `lstat * rm`).
    contrast : str, default="drop"
        - `"drop"`: Uses one-hot encoding, dropping the first observed category.
        - `"sum"`: Uses contrast sum coding (deviation coding).


    Returns:
    X : pd.DataFrame
        The design matrix with categorical variables encoded, formatted, and ordered correctly.
    """

    # 1️⃣ Ensure `df` is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("❌ Error: `df` must be a pandas DataFrame.")

    # 2️⃣ Ensure all specified variables exist in `df`
    missing_vars = [var for var in terms if var not in df.columns]
    if missing_vars:
        raise ValueError(f"❌ Error: The following columns are missing from the DataFrame: {missing_vars}")

    # 3️⃣ Identify categorical and numeric variables
    categorical_vars = [var for var in terms if df[var].dtype.name == "category" or df[var].dtype == "object"]
    numeric_vars = [var for var in terms if pd.api.types.infer_dtype(df[var]) in ['integer', 'floating']]

    # 4️⃣ Determine the first observed category for each categorical variable
    first_seen_categories = {cat_var: df[cat_var].iloc[0] for cat_var in categorical_vars}
    first_alpha_categories = {cat_var: np.sort(df[cat_var].unique())[0] for cat_var in categorical_vars}

    # 5️⃣ Build the formula dynamically, ensuring the first observed category is dropped
    formula_parts = []
    categorical_mapping = {}  # Track categorical variable replacements for ordering
    for var in terms:
        if var in numeric_vars:
            formula_parts.append(var)
        if var in categorical_vars:
            if contrast == "drop":
                first_category = first_alpha_categories[var]  # Drop first alphabetic category
                formula_parts.append(f"C({var}, Treatment('{first_category}'))")
            elif contrast == "sum":
                formula_parts.append(f"C({var}, Sum)")
            else:
                raise ValueError("❌ Error: `contrast` must be 'drop' or 'sum'.")
            


    # 6️⃣ Add interaction terms if provided
    if interactions:
        interaction_terms = []
        for term1, term2 in interactions:
            interaction_terms.append(f"{term1}:{term2}")  # Formulaic uses ":" for interactions
        formula_parts.extend(interaction_terms)

    # 7️⃣ Construct the final formula
    formula = "1 + " + " + ".join(map(str, formula_parts))

    # 8️⃣ Generate the design matrix using Formulaic
    X = formulaic.model_matrix(formula, df)

    # 9️⃣ Rename "Intercept" column to "intercept"
    if "Intercept" in X.columns:
        X = X.rename(columns={"Intercept": "intercept"})

    # 🔟 Rename categorical variable names to "oldvariable[value]" format
    new_col_names = {}
    if contrast == "drop":
        for col in X.columns:
            if "C(" in col and "Treatment" in col:  # Formulaic encodes as "C(variable, Treatment)[T.value]"
                original_var = col.split(",")[0].replace("C(", "").strip()
                category_value = col.split("[T.")[1].replace("]", "").strip()
                new_col_name = f"{original_var.strip()}[{category_value.strip()}]"
                new_col_names[col] = new_col_name
                categorical_mapping.setdefault(original_var.strip(), []).append(new_col_name)  # Track dummy variable names
    elif contrast == "sum":
        for col in X.columns:
            if "C(" in col :  # Formulaic encodes as "C(variable, Treatment)[T.value]"
                original_var = col.split(",")[0].replace("C(", "").strip()
                category_value = col.split("[T.")[1].replace("]", "").strip()
                new_col_name = f"{original_var.strip()}[{category_value.strip()}]"
                new_col_names[col] = new_col_name
                categorical_mapping.setdefault(original_var.strip(), []).append(new_col_name)  # Track dummy variable names
    # Apply renaming
    X = X.rename(columns=new_col_names)

    # 🔹 11. Reorder columns to match `variables` order exactly
    ordered_columns = ["intercept"] if "intercept" in X.columns else []

    for var in terms:
        if var in numeric_vars:
            ordered_columns.append(var)  # Numeric variables remain as is
        elif var in categorical_mapping:
            ordered_columns.extend(categorical_mapping[var])  # Replace categorical column with its dummy variables

    # Add interaction terms at the end
    if interactions:
        for term1, term2 in interactions:
            interaction_col1 = f"{term1}:{term2}"
            interaction_col2 = f"{term2}:{term1}"  # Some formats might flip order
            if interaction_col1 in X.columns:
                ordered_columns.append(interaction_col1)
            elif interaction_col2 in X.columns:
                ordered_columns.append(interaction_col2)

 

    return X


from sklearn.base import (BaseEstimator,RegressorMixin,TransformerMixin)
import pandas as pd

class CustomDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, terms, interactions=None, contrast="drop"):
        self.terms = terms
        self.interactions = interactions
        self.contrast = contrast

    def fit(self, X, y=None):
        # Aucun ajustement nécessaire pour ce transformateur
        return self  

    def transform(self, X):
        # Appliquer la transformation personnalisée
        return transform_df_for_model(X, self.terms, self.interactions, self.contrast)
    
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

# juste pour harmoniser entre sm and sklearn

import numpy as np



from sklearn.utils.validation import check_is_fitted

import statsmodels.api as sm


class sklearn_sm(BaseEstimator,
                 RegressorMixin): 

    """
    Parameters
    ----------

    model_type: class
        A model type from statsmodels, e.g. sm.OLS or sm.GLM

    model_spec: ModelSpec
        Specify the design matrix.

    model_args: dict (optional)
        Arguments passed to the statsmodels model.

    Notes
    -----

    If model_str is present, then X and Y are presumed
    to be pandas objects that are placed
    into a dataframe before formula is evaluated.
    This affects `fit` and `predict` methods.

    """

    def __init__(self,
                 model_type,
                 model_spec=None,
                 model_args={}):

        self.model_type = model_type
        self.model_spec = model_spec
        self.model_args = model_args
        
    def fit(self, X, y):
        """
        Fit a statsmodel model
        with design matrix 
        determined from X and response y.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.
        """

        if self.model_spec is not None:
            self.model_spec_ = self.model_spec.fit(X)
            X = self.model_spec_.transform(X)
        self.model_ = self.model_type(y, X, **self.model_args)
        self.results_ = self.model_.fit()

    def predict(self, X):
        """
        Compute predictions
        for design matrix X.

        Parameters
        ----------

        X : array-like
            Design matrix.

        """
        if self.model_spec is not None:
            X = self.model_spec_.transform(X)
        return self.results_.predict(exog=X)
 
    def score(self, X, y, sample_weight=None):
        """
        Score a statsmodel model
        with test design matrix X and 
        test response y.

        If model_type is OLS, use MSE. For
        a GLM this computes (average) deviance.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.

        sample_weight : None
            Optional sample weights.
        """

        yhat = self.predict(X)
        if isinstance(self.model_, sm.OLS):
            if sample_weight is None:
                return np.mean((y-yhat)**2)
            else:
                return (np.mean((y-yhat)**2*sample_weight) /
                        np.mean(sample_weight))
                
        elif isinstance(self.model_, sm.GLM):
            if sample_weight is None:
                return self.model_.family.deviance(y,
                                                   yhat).mean()
            else:
                value = self.model_.family.deviance(y,
                                                    yhat,
                                                    freq_weights=sample_weight).mean()

                return value / np.mean(sample_weight)

           