reinitialize

# ======================================================================
# INPUT STRUCTURES
# ======================================================================

load data/raw/pdb/3SPU/3SPU.cif, NDM1_raw
load data/raw/pdb/5YD7/5YD7.cif, VIM2_raw

create NDM1, NDM1_raw and chain B
create VIM2, VIM2_raw and chain B

delete NDM1_raw
delete VIM2_raw

# ======================================================================
# VERIFIED STRUCTURAL SELECTIONS
# ======================================================================

select ndm_polymer, NDM1 and polymer
select vim_polymer, VIM2 and polymer

select ndm_catalytic_zn, \
    NDM1 and resn ZN and resi 301+302

select vim_catalytic_zn, \
    VIM2 and resn ZN and resi 301+302

select vim_non_catalytic_zn, \
    VIM2 and resn ZN and resi 303

select ndm_metal_ligands, \
    NDM1 and polymer and resi 120+122+124+189+208+250

select vim_metal_ligands, \
    VIM2 and polymer and resi 114+116+118+179+198+240

# ======================================================================
# VALIDATION
# ======================================================================

python
from pymol import cmd

checks = {
    "NDM polymer CA": cmd.count_atoms("ndm_polymer and name CA"),
    "VIM polymer CA": cmd.count_atoms("vim_polymer and name CA"),
    "NDM catalytic Zn": cmd.count_atoms("ndm_catalytic_zn"),
    "VIM catalytic Zn": cmd.count_atoms("vim_catalytic_zn"),
    "VIM non-catalytic Zn303": cmd.count_atoms("vim_non_catalytic_zn"),
    "NDM metal-ligand residues": cmd.count_atoms(
        "ndm_metal_ligands and name CA"
    ),
    "VIM metal-ligand residues": cmd.count_atoms(
        "vim_metal_ligands and name CA"
    ),
}

print("STRUCTURAL SELECTION CHECKS")
for name, value in checks.items():
    print(f"{name}: {value}")
python end

# ======================================================================
# COMPARATIVE SUPERPOSITION
#
# VIM-2 is transformed onto NDM-1 for visualization only.
# The coordinate files themselves are not modified.
# ======================================================================

align \
    VIM2 and polymer and name CA, \
    NDM1 and polymer and name CA, \
    cycles=5

# ======================================================================
# PUBLICATION RENDER SETTINGS
# ======================================================================

bg_color white

set ray_opaque_background, off
set opaque_background, off

set antialias, 2
set ray_trace_mode, 0
set ray_shadows, 1
set depth_cue, 0
set fog, 0

# Reduce the glossy appearance seen in the diagnostic.
set specular, 0.45
set shininess, 35
set ambient, 0.35
set direct, 0.80
set reflect, 0.20

set cartoon_smooth_loops, 1
set cartoon_fancy_helices, 1
set cartoon_flat_sheets, 1

set sphere_scale, 0.42
set stick_radius, 0.15

# ======================================================================
# COLORS
# ======================================================================

set_color ndm_cream, [0.933, 0.863, 0.604]
set_color vim_coral, [0.886, 0.604, 0.498]
set_color zinc_gold, [0.94, 0.64, 0.18]
set_color catalytic_carbon, [0.78, 0.32, 0.24]

color ndm_cream, ndm_polymer
color vim_coral, vim_polymer

color zinc_gold, ndm_catalytic_zn
color zinc_gold, vim_catalytic_zn

# ======================================================================
# COMMON CAMERA
#
# Determine one camera from the superposed homologues.
# ======================================================================

hide everything, all

show cartoon, ndm_polymer
show cartoon, vim_polymer
show spheres, ndm_catalytic_zn
show spheres, vim_catalytic_zn

orient ndm_polymer or vim_polymer
zoom ndm_polymer or vim_polymer, 4

turn y, -8
turn x, 5

python
from pymol import cmd
common_view = cmd.get_view()
python end

# ======================================================================
# NDM-1 OVERVIEW
# ======================================================================

hide everything, all

show cartoon, ndm_polymer
show spheres, ndm_catalytic_zn

color ndm_cream, ndm_polymer
color zinc_gold, ndm_catalytic_zn

python
from pymol import cmd
cmd.set_view(common_view)
python end

ray 3600, 3000
png figures/components/pymol/NDM1_3SPU_chainB_overview.png, dpi=600

# ======================================================================
# VIM-2 OVERVIEW
# ======================================================================

hide everything, all

show cartoon, vim_polymer
show spheres, vim_catalytic_zn

color vim_coral, vim_polymer
color zinc_gold, vim_catalytic_zn

python
from pymol import cmd
cmd.set_view(common_view)
python end

ray 3600, 3000
png figures/components/pymol/VIM2_5YD7_chainB_overview.png, dpi=600

# ======================================================================
# NDM-1 CATALYTIC VIEW
# ======================================================================

hide everything, all

show cartoon, ndm_polymer
show spheres, ndm_catalytic_zn
show sticks, ndm_metal_ligands

color ndm_cream, ndm_polymer
color zinc_gold, ndm_catalytic_zn

color catalytic_carbon, ndm_metal_ligands and elem C
color nitrogen, ndm_metal_ligands and elem N
color oxygen, ndm_metal_ligands and elem O
color sulfur, ndm_metal_ligands and elem S

python
from pymol import cmd
cmd.set_view(common_view)
python end

ray 3600, 3000
png figures/components/pymol/NDM1_3SPU_chainB_catalytic.png, dpi=600

# ======================================================================
# VIM-2 CATALYTIC VIEW
# ======================================================================

hide everything, all

show cartoon, vim_polymer
show spheres, vim_catalytic_zn
show sticks, vim_metal_ligands

color vim_coral, vim_polymer
color zinc_gold, vim_catalytic_zn

color catalytic_carbon, vim_metal_ligands and elem C
color nitrogen, vim_metal_ligands and elem N
color oxygen, vim_metal_ligands and elem O
color sulfur, vim_metal_ligands and elem S

python
from pymol import cmd
cmd.set_view(common_view)
python end

ray 3600, 3000
png figures/components/pymol/VIM2_5YD7_chainB_catalytic.png, dpi=600

quit
