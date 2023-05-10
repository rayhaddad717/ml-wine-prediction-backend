from voluptuous import Schema, Required, All, Coerce, Range

wine_quality_schema = Schema({
    Required('fixedAcidity'): All(Coerce(float), Range(min=0)),
    Required('volatileAcidity'): All(Coerce(float), Range(min=0)),
    Required('citricAcid'): All(Coerce(float), Range(min=0)),
    Required('residualSugar'): All(Coerce(float), Range(min=0)),
    Required('chlorides'): All(Coerce(float), Range(min=0)),
    Required('freeSulfurDioxide'): All(Coerce(float), Range(min=0)),
    Required('density'): All(Coerce(float), Range(min=0)),
    Required('pH'): All(Coerce(float), Range(min=0)),
    Required('sulphates'): All(Coerce(float), Range(min=0)),
    Required('alcohol'): All(Coerce(float), Range(min=0)),
})

# when reading data from csv (key titles have spaces)
wine_quality_schema_csv = Schema({

    Required('fixed acidity'): All(Coerce(float), Range(min=0)),
    Required('volatile acidity'): All(Coerce(float), Range(min=0)),
    Required('citric acid'): All(Coerce(float), Range(min=0)),
    Required('residual sugar'): All(Coerce(float), Range(min=0)),
    Required('chlorides'): All(Coerce(float), Range(min=0)),
    Required('free sulfur dioxide'): All(Coerce(float), Range(min=0)),
    Required('density'): All(Coerce(float), Range(min=0)),
    Required('pH'): All(Coerce(float), Range(min=0)),
    Required('sulphates'): All(Coerce(float), Range(min=0)),
    Required('alcohol'): All(Coerce(float), Range(min=0)),
})