
import importlib

from fastapi import (FastAPI, Request)
from fastapi.responses import JSONResponse

from configloader import (server_config)
from shared import ATMError


'''
    1. App Initializations
    Description: 
        1. Import procedures specified in `server_config.setup.engine_id_list`
        2. Add new router to FastAPI app instance using `fastapi.app.include_router`
'''

app = FastAPI()

@app.exception_handler(ATMError)
async def unicorn_exception_handler(request: Request, exc: ATMError):
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__,
                 "message": exc.args[0]},
    )



for engine_id in server_config['setup']['engine_id_list']:
    try:
        engine_module = importlib.import_module(f"engines.{engine_id.replace('-', '_')}")
        app.include_router(engine_module.router) #, prefix=f"/")
    except Exception as exc:
        raise
        