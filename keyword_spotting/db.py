from pony import orm
from datetime import datetime 


class Model(orm.db.Entity):
    class_name = orm.Required(str)
    params = orm.Required(dict)    
    hash = orm.Required(str)
    created_at =  orm.Required(datetime.datetime,
                          default=datetime.datetime.utcnow)


class PreProcessing(orm.db.Entity):
    description = orm.Required(str)


class Result(orm.db.Entity):
    model = orm.Required(Model)
    metrics = orm.Required(dict)
    preprocessing = orm.Required(PreProcessing)
    

def get_or_create_model(model):
    hash = model.hash()
    if not Model.get(hash):
        model_db = Model(class_name=type(model).__name__ ,
                          parms=model.to_json(),
                           hash=hash)
        model_db.save()
    else:
        model_db
    return model_db

def get_or_create_preprocessing(preprocessing):
    pass 

def store_test(preprocessing, model, result):
    model_db = get_or_create_model(model)            
    preprocessing_db = get_or_create_preprocessing(preprocessing) 

    result = Result(model=model_db, preprocessing=preprocessing_db, metrics=result)
    result.save()
    