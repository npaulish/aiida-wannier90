#!/usr/bin/env python
################################################################################
# Copyright (c), AiiDA team and individual contributors.                       #
#  All rights reserved.                                                        #
# This file is part of the AiiDA-wannier90 code.                               #
#                                                                              #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-wannier90 #
# For further information on the license, see the LICENSE.txt file             #
################################################################################
"""Example script to launch a MinimalW90WorkGraph for GaAs."""
from aiida import load_profile, orm, tools

from aiida_wannier90.orbitals import generate_projections
from aiida_wannier90.workflows.minimal_workgraph import minimal_w90_workgraph

load_profile()

structure = orm.load_node(104)  # GaAs

# KpointsData = orm.DataFactory('core.array.kpoints')
kpoints_mesh = orm.KpointsData()
kpoints_mesh.set_kpoints_mesh([2, 2, 2])

kpath_parameters = tools.data.array.kpoints.main.get_kpoints_path(structure)
kpath = kpath_parameters["parameters"]
structure = kpath_parameters["primitive_structure"]

# sp^3 projections, centered on As
ase = structure.get_ase()
cell = ase.get_cell()
a = cell[0][1]
projections = generate_projections(
    {
        "position_cart": (-a / 2.0, a / 2.0, a / 2.0),
        "ang_mtm_l_list": [-3],
        "spin": None,
        "spin_axis": None,
    },
    structure=structure,
)

inputs = {
    "scf": {"kpoints": kpoints_mesh},
    "nscf": {"kpoints": kpoints_mesh},
    "wannier90": {
        "kpoint_path": kpath,
        "projections": projections,
    },
}
wg = minimal_w90_workgraph(
    codes={
        "pw": "qe-7.4-pw@localhost",
        "pw2wannier90": "qe-7.4-pw2wannier90@localhost",
        "wannier90": "wannier90@localhost",
    },
    structure=structure,
    pseudo_family="SSSP/1.3/PBE/efficiency",
    inputs=inputs,
)

# ------------------------- Submit the calculation -------------------
wg.submit()
