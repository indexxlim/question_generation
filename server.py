
import importlib

from fastapi import (FastAPI, Request)
from fastapi.responses import JSONResponse

from configloader import server_config
server_config = server_config['server_configs']

'''
    1. App Initializations
    Description: 
        1. Import procedures specified in `server_config.setup.engine_id_list`
        2. Add new router to FastAPI app instance using `fastapi.app.include_router`
'''

app = FastAPI()

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__,
                 "message": exc.args[0]},
    )



try:
    engine_module = importlib.import_module(f"inference")
    app.include_router(engine_module.router) #, prefix=f"/")
except Exception as exc:
    raise
    