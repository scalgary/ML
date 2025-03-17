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