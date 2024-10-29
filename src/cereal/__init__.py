import os
import capnp
from pathlib import Path

devel_path = Path(os.path.dirname(os.path.abspath(__file__)))
aw_mini_path = devel_path.parents[4]
CEREAL_PATH = aw_mini_path / Path("src/autoware_mini/src/cereal") 
capnp.remove_import_hook()

log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))
