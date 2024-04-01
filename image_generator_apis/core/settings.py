from functools import lru_cache
from pathlib import Path
from .envs import Envs

@lru_cache
def get_envs() -> Envs :
    """get environmental values

    Usage:
    ``` python
    @app.get('/')
    async def sample_api(envs: Annotated[Envs, Depends(get_envs)]) :
        return {myvar: envs.myvar}
    ```

    Returns:
        Envs: pydantic object-like environmental values

    Extras:   
        This function is cached
    """
    return Envs()

# From the file
BASE_DIR = Path(__file__).parent.parent
"""The base directory of the project"""