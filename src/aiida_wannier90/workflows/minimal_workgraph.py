"""A minimal WorkGraph to run Wannier90."""

from copy import deepcopy

from aiida_workgraph import WorkGraph, task

from aiida import orm

from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_quantumespresso.calculations.pw import PwCalculation

from aiida_wannier90.calculations.wannier90 import Wannier90Calculation


@task.calcfunction()
def get_explicit_kpoints(kpoints: orm.KpointsData) -> orm.KpointsData:
    """Convert from a mesh to an explicit list."""

    kpt = orm.KpointsData()
    kpt.set_kpoints(kpoints.get_kpoints_mesh(print_list=True))
    return kpt


def minimal_w90_workgraph(  # pylint: disable=too-many-statements
    codes: dict,
    structure: orm.StructureData,
    pseudo_family: str,
    inputs: dict = None,
) -> WorkGraph:
    """Minimal WorkGraph to run Quantum ESPRESSO + Wannier90 for GaAs.

    Mostly to be used as an example, as there is no
    error checking and runs directly Quantum ESPRESSO calculations rather
    than the base workflows.
    """

    for k, v in codes.items():
        if isinstance(v, str):
            codes[k] = orm.load_code(v)

    pseudo_family = orm.load_group(pseudo_family)
    pseudos = pseudo_family.get_pseudos(structure=structure)

    metadata_default_mpi = {
        "options": {
            "resources": {"num_machines": 2},
            "max_wallclock_seconds": 60 * 60 * 12,
            "withmpi": True,
        }
    }
    metadata_default_no_mpi = {
        "options": {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 60 * 60 * 12,
            "withmpi": False,
        }
    }

    # set up the scf parameters input dictionary
    if inputs is None:
        inputs = {}
    if "scf" not in inputs.keys():
        inputs["scf"] = {}
    if "parameters" not in inputs["scf"].keys():
        inputs["scf"]["parameters"] = {}
    if "CONTROL" not in inputs["scf"]["parameters"].keys():
        inputs["scf"]["parameters"]["CONTROL"] = {}
    if "SYSTEM" not in inputs["scf"]["parameters"].keys():
        inputs["scf"]["parameters"]["SYSTEM"] = {}

    ecutwfc = inputs["scf"]["parameters"]["SYSTEM"].get("ecutwfc", 30.0)
    ecutrho = inputs["scf"]["parameters"]["SYSTEM"].get("ecutwfc", 30.0) * 8.0

    inputs["scf"]["parameters"]["CONTROL"].update(
        {
            "calculation": "scf",
        }
    )
    inputs["scf"]["parameters"]["SYSTEM"].update(
        {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
        }
    )

    scf_metadata = deepcopy(metadata_default_mpi)
    scf_metadata.update(inputs["scf"].get("metadata", {}))

    # set up the nscf parameters input dictionary
    assert (
        "nscf" in inputs.keys()
    ), "nscf parameters not specified, please provide at least kpoints"
    assert (
        "kpoints" in inputs["nscf"].keys()
    ), "please provide kpoint mesh for the nscf calculation"
    nscf_kpoints = None
    try:
        inputs["nscf"]["kpoints"].get_kpoints()
        raise ValueError(
            "nscf kpoints should be specified as an MP grid, \
                         it will be converted to as explicit mesh automaticaly"
        )
    except AttributeError:
        assert inputs["nscf"]["kpoints"].get_kpoints_mesh()[1] == [
            0,
            0,
            0,
        ], "k-point mesh for the nscf calculation should be unshifted"
        nscf_kpoints = get_explicit_kpoints(inputs["nscf"]["kpoints"])

    if "parameters" not in inputs["nscf"].keys():
        inputs["nscf"]["parameters"] = {}
    if "CONTROL" not in inputs["nscf"]["parameters"].keys():
        inputs["nscf"]["parameters"]["CONTROL"] = {}
    if "SYSTEM" not in inputs["scf"]["parameters"].keys():
        inputs["nscf"]["parameters"]["SYSTEM"] = deepcopy(
            inputs["scf"]["parameters"]["SYSTEM"]
        )

    inputs["nscf"]["parameters"]["CONTROL"].update(
        {
            "calculation": "nscf",
        }
    )
    # check if ecutrho and ecutwfc were specified for nscf by the user?

    nscf_metadata = deepcopy(metadata_default_mpi)
    nscf_metadata.update(inputs["nscf"].get("metadata", {}))

    # set up pw2wannier90 parameters

    pw2wannier_parameters_default = {
        "inputpp": {
            "write_amn": True,
            "write_unk": True,
            "write_mmn": True,
        }
    }

    if "pw2wannier90" not in inputs.keys():
        inputs["pw2wannier90"] = {}
    pw2wannier_settings = inputs["pw2wannier90"].get("settings", {})
    pw2wannier_settings.update(
        {"ADDITIONAL_RETRIEVE_LIST": ["*.amn", "*.mmn", "*.eig"]}
    )
    pw2wannier90_params = deepcopy(inputs["pw2wannier90"].get("parameters", {}))
    inputpp = deepcopy(pw2wannier_parameters_default["inputpp"])
    inputpp.update(pw2wannier90_params.get("inputpp", {}))
    pw2wannier90_params["inputpp"] = inputpp

    pw2wan_metadata = deepcopy(metadata_default_mpi)
    pw2wan_metadata.update(inputs["pw2wannier90"].get("metadata", {}))

    # set w90_pp and w90 parameters
    if "wannier90" not in inputs.keys():
        inputs["wannier90"] = {}

    w90_params_default = {
        "mp_grid": inputs["nscf"]["kpoints"].get_kpoints_mesh()[0],
        "write_hr": False,
        "write_xyz": False,
        "use_ws_distance": True,
        "bands_plot": True,
        "num_iter": 200,
        "guiding_centres": False,
        "num_wann": 4,
        "exclude_bands": [1, 2, 3, 4, 5],
    }
    w90_params = deepcopy(w90_params_default)
    w90_params.update(inputs["wannier90"].get("parameters", {}))

    wannier90_metadata = deepcopy(metadata_default_no_mpi)
    wannier90_metadata.update(inputs["wannier90"].get("metadata", {}))

    wg = WorkGraph("Minimal Wannier90 workchain")
    scf_task = wg.add_task(PwCalculation, name="scf")
    scf_task.set(
        {
            "code": codes["pw"],
            "structure": structure,
            "pseudos": pseudos,
            "parameters": orm.Dict(inputs["scf"]["parameters"]),
            "kpoints": inputs["scf"]["kpoints"],
            "metadata": scf_metadata,
        }
    )

    nscf_task = wg.add_task(PwCalculation, name="nscf")
    nscf_task.set(
        {
            "code": codes["pw"],
            "structure": structure,
            "pseudos": pseudos,
            "parameters": orm.Dict(inputs["nscf"]["parameters"]),
            "kpoints": nscf_kpoints,
            "parent_folder": scf_task.outputs["remote_folder"],
            "metadata": nscf_metadata,
        }
    )

    w90_pp_task = wg.add_task(Wannier90Calculation, name="w90_pp")
    w90_pp_task.set(
        {
            "code": codes["wannier90"],
            "structure": structure,
            "parameters": orm.Dict(w90_params),
            "kpoints": nscf_kpoints,  # why do I need to specify both kpoints here and MP grid in w90_pp_parameters?
            "kpoint_path": inputs["wannier90"]["kpoint_path"],
            "projections": inputs["wannier90"]["projections"],
            "settings": orm.Dict({"postproc_setup": True}),
            "metadata": metadata_default_no_mpi,
        }
    )
    pw2wan_task = wg.add_task(Pw2wannier90Calculation, name="pw2wan")
    pw2wan_task.set(
        {
            "code": codes["pw2wannier90"],
            "parameters": orm.Dict(pw2wannier90_params),
            "parent_folder": nscf_task.outputs["remote_folder"],
            "nnkp_file": w90_pp_task.outputs["nnkp_file"],
            "settings": orm.Dict(pw2wannier_settings),
            "metadata": pw2wan_metadata,
        }
    )
    w90_task = wg.add_task(Wannier90Calculation, name="w90")
    w90_task.set(
        {
            "code": codes["wannier90"],
            "structure": structure,
            "parameters": orm.Dict(w90_params),
            "kpoints": nscf_kpoints,
            "kpoint_path": inputs["wannier90"]["kpoint_path"],
            "remote_input_folder": pw2wan_task.outputs["remote_folder"],
            "projections": inputs["wannier90"]["projections"],
            "metadata": wannier90_metadata,
        }
    )
    return wg
