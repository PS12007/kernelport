import copy
import json

import pytest

from conftest import ZOO
from kernelport.harness.spec import LaunchSpec

BASE = json.loads((ZOO / "vector_add" / "launch.json").read_text())


def test_all_zoo_specs_load():
    for d in ZOO.iterdir():
        if (d / "launch.json").exists():
            LaunchSpec.load(d / "launch.json")


def test_entry_template():
    spec = LaunchSpec.from_dict(BASE)
    assert spec.entry_for("float16") == "vector_add<__half>"


def test_grid_block_padding():
    spec = LaunchSpec.from_dict(BASE)
    assert spec.grid_block({"n": 1000}) == ((4, 1, 1), (256, 1, 1), 0)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["params"].append(dict(d["params"][0])), "duplicate"),
    (lambda d: d["params"][0].update(kind="weird"), "kind"),
    (lambda d: d.update(dtypes=["float8"]), "unknown dtype"),
    (lambda d: d.update(cases=[]), "no cases"),
    (lambda d: d["init"].update(zz={}), "unknown param"),
    (lambda d: d.update(params=[p for p in d["params"] if p["kind"] != "output"]), "no output"),
    (lambda d: d["cases"][0]["dims"].pop("n"), "unknown dimension"),
])
def test_validation_errors(mutate, msg):
    d = copy.deepcopy(BASE)
    mutate(d)
    with pytest.raises(ValueError, match=msg):
        LaunchSpec.from_dict(d)


def test_hash_is_stable():
    assert LaunchSpec.from_dict(BASE).hash() == LaunchSpec.from_dict(copy.deepcopy(BASE)).hash()
