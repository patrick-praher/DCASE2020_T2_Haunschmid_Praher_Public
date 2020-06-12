type_metric_mapping = {"ToyCar": "50%", "ToyConveyor": "50%", "fan": "50%",
                       "pump": "50%", "slider": "std", "valve": "std"}


def postprocess(snippets, metric=None):
    if metric is None:
        metric = 'mean'

    snippets = snippets.melt(id_vars=['ID'])
    agg = snippets.groupby(["ID"]).describe().reset_index()
    agg.columns = agg.columns = [' '.join(col).strip() for col in agg.columns.values]
    predictions = agg["value " + metric].tolist()
    return predictions