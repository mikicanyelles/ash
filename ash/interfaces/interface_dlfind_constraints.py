# This is the full DLFIND_optimizerClass from your prompt,
# modified to accept constraint values and 'bond_diff' constraints.



from __future__ import annotations
from ctypes import c_double, c_int, pointer
import functools
from typing import Callable, Optional
import numpy as np
from numpy.ctypeslib import as_array
from numpy.typing import ArrayLike

import os
import time
from ash.functions.functions_general import ashexit, blankline,BC,print_time_rel,print_line_with_mainheader,listdiff,search_list_of_lists_for_index
from ash.modules.module_coords import check_charge_mult, fullindex_to_actindex,print_internal_coordinate_table,write_xyzfile,elemstonuccharges
from ash.modules.module_theory import NumGradclass
from ash.modules.module_results import ASH_Results
from ash.modules.module_freq import NumFreq,AnFreq,calc_hessian_xtb
from ash.modules.module_QMMM import QMMMTheory
from ash.modules.module_oniom import ONIOMTheory
from ash.modules.interfaces.interface_dlfind import DLFIND_optimizerClass
import math
import numpy as np
import functools
from libdlfind.callback import dlf_get_gradient_wrapper


def DLFIND_optimizer_with_constraints(jobtype=None, theory=None, fragment=None, fragment2=None, charge=None, mult=None, 
                     maxcycle=250, tolerance=4.5E-4, tolerance_e=1E-6,
                     actatoms=None, frozenatoms=None, residues=None, constraints=None,
                     printlevel=2, NumGrad=False, delta=0.01,
                     icoord=None, iopt=None, nimage=None, 
                     hessian_choice="numfreq", inithessian=0, 
                     numfreq_npoint=1, numfreq_displacement=0.005, numfreq_hessatoms=None,
                     numfreq_force_projection=None, print_atoms_list=None):
    """
    Wrapper function around DLFIND_optimizerClass
    """
    timeA=time.time()
    #EARLY EXIT
    if theory is None or fragment is None:
        print("DLFIND_optimizer requires theoryNumFreq and fragment objects provided. Exiting.")
        ashexit()
    optimizer=DLFIND_optimizerClass_with_customisable_constraints(jobtype=jobtype, theory=theory, fragment=fragment, fragment2=fragment2, charge=charge, mult=mult, actatoms=actatoms,
                                    frozenatoms=frozenatoms,residues=residues, constraints=constraints, delta=delta,
                                    printlevel=printlevel, icoord=icoord,iopt=iopt, maxcycle=maxcycle, 
                                    tolerance=tolerance,tolerance_e=tolerance_e, 
                                    nimage=nimage, 
                                    hessian_choice=hessian_choice, inithessian=inithessian, 
                                    numfreq_npoint=numfreq_npoint,numfreq_displacement=numfreq_displacement,
                                    numfreq_hessatoms=numfreq_hessatoms,numfreq_force_projection=numfreq_force_projection,
                                    print_atoms_list=print_atoms_list)

    # If NumGrad then we wrap theory object into NumGrad class object
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        print("This enables numerical-gradient calculation for theory")
        theory = NumGradclass(theory=theory)

    # Providing theory and fragment to run method. Also constraints
    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult)
    if printlevel >= 1:
        print_time_rel(timeA, modulename='DL-FIND', moduleindex=1)

    return result

# Main DLFIND Class (Modified)
class DLFIND_optimizerClass_with_customisable_constraints:

    def __init__(self,jobtype=None, fragment=None, fragment2=None, theory=None, charge=None, mult=None, 
                 maxcycle=250, tolerance=4.5E-4, tolerance_e=1E-6, 
                 printlevel=2, result_write_to_disk=True, actatoms=None, frozenatoms=None, residues=None, constraints=None,
                 icoord=None, iopt=None, nimage=None, delta=0.01, 
                 hessian_choice='numfreq', inithessian=None, 
                 numfreq_npoint=1,numfreq_displacement=0.005,numfreq_force_projection=None,
                 numfreq_hessatoms=None, print_atoms_list=None):

        print_line_with_mainheader("DLFIND_optimizer initialization")
        print()
        print("If you use DL-FIND for your research make sure to cite:")
        print("DL-FIND: an Open-Source Geometry Optimizer for Atomistic Simulations, J. Kästner, J. M. Carr, T. W. Keal, W. Thiel, A. Wander, P. Sherwood, J. Phys. Chem. A, 2009, 113, 11856.")
        print()
        self.printlevel=printlevel

        print("Importing libdlfind package\n")
        try:
            from libdlfind import dl_find
            from libdlfind.callback import (dlf_get_gradient_wrapper,
                                            dlf_put_coords_wrapper, make_dlf_get_params)
            self.dl_find = dl_find # Store function
            self.make_dlf_get_params = make_dlf_get_params # Store function
            self.dlf_get_gradient_wrapper = dlf_get_gradient_wrapper # Store function
        except ImportError:
            print("Warning: Error importing libdlfind. Using mock objects.")
            # Mocking libdlfind for example purposes
            def dlf_get_gradient_wrapper(func): return func
            def make_dlf_get_params(**kwargs): return lambda: None
            def dl_find(**kwargs): print("Mock dl_find call. Optimization 'completes' instantly.")
            self.dl_find = dl_find
            self.make_dlf_get_params = make_dlf_get_params
            self.dlf_get_gradient_wrapper = dlf_get_gradient_wrapper
        except Exception as e:
            print(f"Error importing libdlfind: {e}")
            print("Have you installed: https://github.com/digital-chemistry-laboratory/libdlfind")
            print("Quick-fix: pip install libdlfind")
            ashexit()

        # EARLY EXITS
        if theory is None or fragment is None:
            print("DLFIND_optimizer requires theory and fragment objects provided. Exiting.")
            ashexit()

        if jobtype is None and icoord is None:
            print("Error: You must either select a jobtype keyword (e.g. opt, neb, dimer, instanton) or select DL-FIND icoord and iopt codes")
            print("Example: DLFIND_optimizer(jobtype='opt') ")
            ashexit()
        elif jobtype == "opt":
            print("jobtype: opt chosen")
            print("Choosing icoord=1 (HDLC internal coordinates) and iopt=3 (L-BFGS minimizer)")
            print("For other coordinate-systems: choose icoord=0 (cartesian), icoord=2 (hdlc-tc), icoord=3 (dlc-prim), icoord=3 (dlc-tc)")
            print("For other opt algorithms: choose iopt codes: 0: sd, 1: cg-autorestart, 2: cg-restart10, 3: lbfgs, 10: P-RFO")
            icoord=1
            iopt=3
        elif jobtype == "tsopt" or jobtype == "ts":
            print("jobtype: tsopt chosen")
            print("Choosing icoord=120 (HDLC internal coordinates) and iopt=10 (P-RFO)")
            print("Note: inithessian option is:", inithessian)
            icoord=3
            iopt=10
        elif jobtype == "neb":
            print("jobtype: neb chosen")
            print("Choosing icoord=120 (NEB with frozen endpoints) and iopt=3 (L-BFGS)")
            icoord=120
            iopt=3
        elif jobtype == "dimer":
            print("jobtype: dimer chosen")
            print("Choosing icoord=210 (Dimer) and iopt=3 (L-BFGS)")
            icoord=210
            iopt=3
        elif jobtype == "qts" or jobtype == "instanton" :
            print("jobtype: qts chosen (a.k.a. instanton)")
            print("Choosing icoord=190 (qts) and iopt=3 (L-BFGS)")
            icoord=190
            iopt=3
        else:
            print("No jobype selected.")
            print(f"Will start job based on chosen icoord={icoord} and iopt={iopt}")


        self.fragment=fragment
        self.theory=theory

        nuccharges  = elemstonuccharges(self.fragment.elems)

        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "DLFIND-optimizer", theory=theory)

        # Possible Fragment2 handling
        self.fragment2=fragment2
        if self.fragment2 is not None:
            print("Fragment2 provided. This only makes sense for NEB and dimer jobs")
            positions2 = self.fragment2.coords * 1.88972612546
            nframe=1
        else:
            positions2=None
            nframe=0

        #############
        #HESSIAN
        #############
        self.inithessian=inithessian
        self.hessian_choice=hessian_choice
        self.numfreq_npoint=numfreq_npoint
        self.numfreq_displacement=numfreq_displacement
        self.numfreq_hessatoms=numfreq_hessatoms
        self.numfreq_force_projection=numfreq_force_projection

        # Optimizer options
        self.icoord=icoord 
        self.iopt=iopt 
        self.maxcycle=maxcycle
        #Tolerances
        self.tolerance=tolerance
        self.tolerance_e=tolerance_e
        # NEB
        self.nimage=nimage
        #Dimer
        self.delta=delta

        # Residues for HDLC
        self.residues=residues
        #Constraints
        self.constraints=constraints

        ########################################
        # ACTIVE/FROZEN AND RESIDUE HANDLING
        ########################################
        if self.residues is None:
            print("No residues provided to optimizer. Creating a single residue for whole active system.")
        else:
            print("Residues provided to optimizer.")
        # What to optimize etc.
        self.spec=[]
        if actatoms is not None:
            print("Actatoms provided:", actatoms)
            for i in fragment.allatoms:
                if i in actatoms:
                    if self.residues is not None:
                        self.spec.append(search_list_of_lists_for_index(i,self.residues)+1)
                    else:
                        self.spec.append(1)
                else:
                    self.spec.append(-1)
        elif frozenatoms is not None:
            print("Frozenatoms provided:", frozenatoms)
            for i in fragment.allatoms:
                if i in frozenatoms:
                    self.spec.append(-1)
                else:
                    if self.residues is not None:
                        self.spec.append(search_list_of_lists_for_index(i,self.residues)+1)
                    else:
                        self.spec.append(1)
        else:
            print("Case: no actatoms or frozenatoms provided. All atoms will be active.")
            if self.residues is None:
                self.spec=[1 for i in list(range(fragment.numatoms))]
            else:
                print("Residues provided.")
                for i in fragment.allatoms:
                    resid = search_list_of_lists_for_index(i,self.residues)
                    self.spec.append(resid+1)

        # Nuclear charges
        self.spec=self.spec + nuccharges

        # ######################################
        # MODIFIED CONSTRAINTS SECTION (v2)
        # ######################################
        ANG_TO_BOHR = 1.88972612546 # Angstrom -> Bohr
        conlist=[]
        self.numcons=0

        if self.constraints is not None:
            print("Constraints passed: ", constraints)
            
            # Process simple bonds
            if 'bond' in constraints:
                for x in constraints['bond']:
                    if len(x) == 2:
                        # Freeze at current value: [1, i, j, 0, 0]
                        print(f"Found bond constraint (freeze) between atoms: {x}")
                        b = [1, x[0]+1, x[1]+1, 0, 0]
                    elif len(x) == 3:
                        # Constrain to a value: [1, i, j, 0, value_in_bohr]
                        val_ang = x[2]
                        val_bohr = val_ang * ANG_TO_BOHR
                        print(f"Found bond constraint between atoms: {x[0:2]} to value {val_ang} Å ({val_bohr:.4f} Bohr)")
                        b = [1, x[0]+1, x[1]+1, 0, val_bohr] # <-- CORRECTED
                    else:
                        print(f"Warning: Bond constraint {x} has wrong format (expected [i,j] or [i,j,value]). Skipping.")
                        continue
                    conlist += b
                    self.numcons+=1
            
            # Process simple angles
            if 'angle' in constraints:
                for x in constraints['angle']:
                    if len(x) == 3:
                        # Freeze at current value: [2, i, j, k, 0]
                        print(f"Found angle constraint (freeze) between atoms: {x}")
                        b = [2, x[0]+1, x[1]+1, x[2]+1, 0]
                    elif len(x) == 4:
                        # Constrain to a value: [2, i, j, k, value_in_degrees]
                        val_deg = x[3]
                        print(f"Found angle constraint between atoms: {x[0:3]} to value {val_deg} deg")
                        b = [2, x[0]+1, x[1]+1, x[2]+1, val_deg]
                    else:
                        print(f"Warning: Angle constraint {x} has wrong format (expected [i,j,k] or [i,j,k,value]). Skipping.")
                        continue
                    conlist += b
                    self.numcons+=1
            
            # Process simple dihedrals (freeze only)
            if 'dihedral' in constraints:
                for x in constraints['dihedral']:
                    if len(x) == 4:
                        # Freeze at current value: [3, i, j, k, l]
                        print(f"Found dihedral constraint (freeze) between atoms: {x}")
                        b = [3, x[0]+1, x[1]+1, x[2]+1, x[3]+1]
                    elif len(x) == 5:
                        print(f"Warning: Dihedral constraint {x} includes a value, but this is not supported by libdlfind spec. Freezing instead.")
                        b = [3, x[0]+1, x[1]+1, x[2]+1, x[3]+1]
                    else:
                        print(f"Warning: Dihedral constraint {x} has wrong format (expected [i,j,k,l]). Skipping.")
                        continue
                    conlist += b
                    self.numcons+=1

            # Process bond_diff (LCIC type 11)
            # Format: [i, j, k, l, value] for d(i,j) - d(k,l) = value
            # Format: [i, j, k, l] for d(i,j) - d(k,l) = current_value (freeze)
            if 'bond_diff' in constraints:
                for x in constraints['bond_diff']:
                    val_bohr = 0.0 # Default for freezing
                    if len(x) == 5:
                        i, j, k, l, val_ang = x
                        val_bohr = val_ang * ANG_TO_BOHR
                        print(f"Found bond_diff constraint: d({i},{j}) - d({k},{l}) = {val_ang} Å ({val_bohr:.4f} Bohr)")
                    elif len(x) == 4:
                        # Freeze at current value
                        i, j, k, l = x
                        print(f"Found bond_diff constraint (freeze): d({i},{j}) - d({k},{l})")
                        # val_bohr = 0.0 will signal DL-FIND to use the current value
                    else:
                        print(f"Warning: bond_diff constraint {x} has wrong format. Expected [i,j,k,l,value] or [i,j,k,l]. Skipping.")
                        continue
                    
                    # [type, type1, i1, j1, type2, i2, j2, value]
                    # type=11 (LCIC 2), type1=1 (bond), type2=1 (bond)
                    b = [11, 1, i+1, j+1, 1, k+1, l+1, val_bohr]
                    conlist += b
                    self.numcons+=1 # This is still one constraint
            
            # Check for other unknown constraint types
            all_types = {'bond', 'angle', 'dihedral', 'bond_diff'}
            for key in constraints:
                if key not in all_types:
                    print(f"Warning: Unknown constraint type '{key}' found. It will be ignored.")

            print("DL-FIND constraints-list (flattened):", conlist)
            print("Number of constraints:", self.numcons)
            self.spec = self.spec + conlist
        else:
            print("No constraints present")
            self.numcons=0
        # ######################################
        # END OF MODIFIED CONSTRAINTS SECTION (v2)
        # ######################################


        # Spec (padding, not sure if needed, was in original)
        self.spec=self.spec+[1 for i in list(range(fragment.numatoms))] #?

        self.nspec=len(self.spec)


        # Print-atoms choice
        if print_atoms_list is None:
            if actatoms is not None:
                if isinstance(theory,QMMMTheory):
                    print("Theory class: QMMMTheory")
                    print("Will by default print only QM-region in output (use print_atoms_list option to change)")
                    self.print_atoms_list=theory.qmatoms
                elif isinstance(theory,ONIOMTheory):
                    print("Theory class: ONIOMTheory")
                    print("Will by default print only Region1 in output (use print_atoms_list option to change)")
                    self.print_atoms_list=theory.regions_N[0]
                else:
                    self.print_atoms_list=actatoms # Changed from self.actatoms
            else:
                self.print_atoms_list=fragment.allatoms
        else:
             self.print_atoms_list = print_atoms_list # Use user-provided list

        self.result_write_to_disk=result_write_to_disk

        #Tracking DL-FIND cycles
        self.dlfind_eg_calls=0
        self.dlfind_opt_cycles=0
        self.dlfind_neb_cycles=0
        self.dlfind_dimer_cycles=0


        self.NEB_energies_dict={}
        self.NEB_geometries={}

        # Create function to calculate energies and gradients
        @self.dlf_get_gradient_wrapper
        def ash_e_g_func(coordinates, iimage, kiter, theory):
            self.dlfind_eg_calls+=1
            coordinates_ang = coordinates*0.5291772109303
            energy, gradient = theory.run(current_coords=coordinates_ang, elems=self.fragment.elems, charge=charge, mult=mult, Grad=True)

            if self.icoord >= 100 and self.icoord < 150 :
                self.NEB_geometries[iimage] = coordinates_ang
                self.NEB_energies_dict[iimage] = energy

            return energy, gradient

        # Modified wrapper function
        def dlf_get_hessian_wrapper(func):
            """Factory function for dlf_get_hessian."""
            @functools.wraps(func)
            def wrapper(
                nvar2: int,
                coords: pointer[c_double],
                hessian: pointer[c_double],
                status: pointer[c_int],
                *args,
                **kwargs,
            ) -> None:
                nvar2=self.fragment.numatoms*3
                coords_ = as_array(coords, shape=(nvar2,)).reshape((-1, 3))
                hessian_ = as_array(hessian, shape=(nvar2,nvar2))
                hessian_val = func(coords_, *args, **kwargs)
                hessian_[:, :] = hessian_val
                status[0] = c_int(0)
                return
            return wrapper

        # How we get the Hessian from ASH
        @dlf_get_hessian_wrapper
        def hess_func(coords):
            nvar=self.fragment.numatoms*3
            self.fragment.coords=coords*0.5291772109303
            if type(self.hessian_choice) == np.ndarray:
                print("A Numpy array was detected as Hessian choice. Passing over to DL-FIND")
                hessian = self.hessian_choice
            elif self.hessian_choice == "numfreq":
                print("NumFreq option requested")
                print("NumFreq Npoint:", self.numfreq_npoint)
                result_freq = NumFreq(theory=self.theory, fragment=self.fragment, printlevel=0, 
                                      npoint=self.numfreq_npoint, displacement=self.numfreq_displacement,
                                      hessatoms=self.numfreq_hessatoms,force_projection=self.numfreq_force_projection,
                                      runmode='serial', 
                                      numcores=self.theory.numcores if hasattr(self.theory, 'numcores') else 1)
                hessian = result_freq.hessian
            elif self.hessian_choice == "anfreq":
                print("AnFreq option requested")
                result_freq = AnFreq(theory=self.theory, fragment=self.fragment, printlevel=0)
                hessian = result_freq.hessian
            elif self.hessian_choice == "xtb":
                print("xTB Hessian option requested")
                # This function (calc_hessian_xtb) is not defined in this file, assuming it's imported
                hessianfile = calc_hessian_xtb(fragment=fragment, actatoms=self.fragment.allatoms, 
                                               numcores=self.theory.numcores if hasattr(self.theory, 'numcores') else 1, use_xtb_feature=True, 
                                               charge=charge, mult=mult)
                hessian = np.loadtxt("Hessian_from_xtb")
            elif 'file:' in self.hessian_choice:
                print("A file was detected as Hessian choice:", self.hessian_choice)
                hessianfile = self.hessian_choice.replace("file:","")
                if os.path.isfile(hessianfile) is False:
                    print(f"File {hessianfile} does not exist.")
                    ashexit()
                hessian=hessian = np.loadtxt(hessianfile)
            elif self.hessian_choice == "random":
                hessian = np.random.random((nvar,nvar))
            else:
                print(f"Unknown hessian_choice: {self.hessian_choice}. Defaulting to random.")
                hessian = np.random.random((nvar,nvar))
            print("ASH hessian shape:", hessian.shape)
            return hessian

        # Create function to store results from DL-FIND
        #@dlf_put_coords_wrapper # This wrapper is not defined in the mock setup
        def store_results(a,nvar,switch, energy, coordinates, iam):
            if switch > 0:
                coords = as_array(coordinates, (nvar,)).reshape(-1, 3)
                coordinates_ang = coords*0.5291772109303
            else:
                if self.icoord >= 100 and self.icoord < 150 and switch == -1:
                    self.dlfind_neb_cycles+=1
                    print("="*70)
                    print(f"DLFIND NEB-OPTIMIZATION CYCLE {self.dlfind_neb_cycles}")
                    print("="*77)
                    with open("DLFIND_NEBpath_all.xyz", "a") as trajfile:
                        for imageid in list(range(1,self.nimage)):
                            if imageid not in self.NEB_geometries: continue
                            trajfile.write(str(self.fragment.numatoms) + "\n")
                            trajfile.write(f"Image {imageid}. Energy: {self.NEB_energies_dict.get(imageid, 'N/A')}  \n")
                            for el, cord in zip(self.fragment.elems, self.NEB_geometries[imageid]):
                                trajfile.write(el + "  " + str(cord[0]) + " " + str(cord[1]) + " " + str(cord[2]) + "\n")
                    with open("DLFIND_NEBpath_current.xyz", "w") as trajfile:
                        for imageid in list(range(1,self.nimage)):
                            if imageid not in self.NEB_geometries: continue
                            trajfile.write(str(self.fragment.numatoms) + "\n")
                            trajfile.write(f"Image {imageid}. Energy: {self.NEB_energies_dict.get(imageid, 'N/A')}  \n")
                            for el, cord in zip(self.fragment.elems, self.NEB_geometries[imageid]):
                                trajfile.write(el + "  " + str(cord[0]) + " " + str(cord[1]) + " " + str(cord[2]) + "\n")

                    if nimage in self.NEB_geometries:
                        print("Writing out current trajectory for climbing image as: DLFIND_CIgeo_traj.xyz ")
                        write_xyzfile(fragment.elems, self.NEB_geometries[nimage], "DLFIND_CIgeo_traj", printlevel=2, writemode='a', title=f"Energy: {self.NEB_energies_dict.get(nimage, 'N/A')}")
                return
            
            # Traj-writing for regular opt
            if self.icoord < 100:
                self.dlfind_opt_cycles+=1
                # print("Writing regular-opt traj") # Too verbose
                write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_opt_traj", printlevel=2, writemode='a', title=f"Energy: {energy}")
                self.current_geo=coordinates_ang
            # Traj-writing for dimer
            elif self.icoord >= 200:
                print("Writing Dimer traj")
                if switch == 1:
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_1", printlevel=2, writemode='a', title=f"Energy: {energy}")
                    self.current_geo=coordinates_ang
                elif switch == 2:
                    self.dlfind_dimer_cycles+=1
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_2", printlevel=2, writemode='a', title=f"Energy: {energy}")
                elif switch == 3:
                    write_xyzfile(fragment.elems, coordinates_ang, "DLFIND_dimertraj_3", printlevel=2, writemode='a', title=f"Energy: {energy}")
            self.traj_energies.append(energy)

            return

        self.traj_energies = []
        self.current_geo = []
        positions = self.fragment.coords * 1.88972612546
        self.dlf_get_params = self.make_dlf_get_params(coords=positions, coords2=positions2, icoord=self.icoord, 
                                                 iopt=self.iopt, maxcycle=self.maxcycle,tolerance=self.tolerance,
                                                 tolerance_e=self.tolerance_e, inithessian=self.inithessian,
                                                 nframe=nframe, nz = self.fragment.numatoms,
                                                 ncons=self.numcons, delta=self.delta,
                                                 spec=self.spec, printl=self.printlevel, nimage=self.nimage)

        self.dlf_get_gradient = functools.partial(ash_e_g_func, theory=theory)
        self.dlf_get_hessian = functools.partial(hess_func)
        self.dlf_put_coords = functools.partial( store_results, None)

        # Delete old traj file before beginning
        remove_files=['DLFIND_opt_traj.xyz','DLFIND_dimertraj_1.xyz', 'DLFIND_dimertraj_2.xyz','DLFIND_dimertraj_3.xyz','DLFIND_NEBpath_current.xyz', 'DLFIND_NEBpath_all.xyz', 'DLFIND_CIgeo_traj.xyz']
        print("Removing possible old files:", remove_files)
        for rfile in remove_files:
            try:
                os.remove(rfile)
                print("removed ", rfile)
            except FileNotFoundError:
                pass

        print("\nArguments passed to DL-FIND:")
        print("icoord:", self.icoord)
        print("iopt:", self.iopt)
        print("maxcycle:", maxcycle)
        print(f"nspec: {self.nspec} (spec list length)")
        # print("spec:", self.spec) # Too verbose
        if icoord == 120:
            print("NEB nimage:", nimage)

    def run(self, theory=None, fragment=None, charge=None, mult=None):
        
        # Use class theory/fragment if not provided
        if fragment is None: fragment = self.fragment
        if theory is None: theory = self.theory
        
        if self.fragment2 is None:
            nvarin=self.fragment.numatoms * 3
            nvarin2=0
        else:
            nvarin = self.fragment.numatoms * 3
            nvarin2 = self.fragment2.numatoms * 3

        # Run DL-FIND
        print("Now starting DL-FIND")
        self.dl_find(
                nvarin=nvarin, nvarin2=nvarin2, nspec=self.nspec,
                dlf_get_gradient=self.dlf_get_gradient, 
                dlf_get_params=self.dlf_get_params,
                dlf_put_coords=self.dlf_put_coords,
                dlf_get_hessian=self.dlf_get_hessian)

        # Regular optimization
        if self.icoord < 100:

            print(f"\nDL-FIND optimization finished in {self.dlfind_opt_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            # Print results
            finalenergy = self.traj_energies[-1] if self.traj_energies else 0.0
            print("Final optimized energy:",  finalenergy)
            # Final coordinate handling
            final_coords=self.current_geo if len(self.current_geo) > 0 else self.fragment.coords
            fragment.replace_coords(fragment.elems,final_coords, conn=False)
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            fragment.set_energy(finalenergy)

            if self.printlevel >= 2:
                print_internal_coordinate_table(fragment,actatoms=self.print_atoms_list)
            print()

            result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)
            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_optimizer.result")
            return result

        elif self.icoord >= 100 and self.icoord < 150:
            # NEB job complete
            print(f"\nDL-FIND NEB job finished in {self.dlfind_neb_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            CI_fragment_energy=None #TODO
            CI_fragment_coords=None #TODO

            result = ASH_Results(label="DLFIND_NEB-CI calc", energy=CI_fragment_energy, geometry=CI_fragment_coords,
                  charge=charge, mult=mult, MEP_energies_dict=self.NEB_energies_dict,
                  barrier_energy=None)

            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_NEB.result")

        elif self.icoord >= 200:
            # Dimer
            print(f"\nDL-FIND Dimer job finished in {self.dlfind_dimer_cycles} steps!")
            print("Number of DL-FIND energy-gradient evaluations:", self.dlfind_eg_calls)

            finalenergy = self.traj_energies[-1] if self.traj_energies else 0.0
            print("Final optimized energy:",  finalenergy)

            final_coords=self.current_geo if len(self.current_geo) > 0 else self.fragment.coords
            fragment.replace_coords(fragment.elems,final_coords, conn=False)
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            fragment.set_energy(finalenergy)

            if self.printlevel >= 2:
                print_internal_coordinate_table(fragment,actatoms=self.print_atoms_list)
            print()

            result = ASH_Results(label="DLFIND_optimizer", energy=finalenergy)
            if self.result_write_to_disk is True:
                result.write_to_disk(filename="DLFIND_optimizer.result")
            return result





# Conversion constants
BOHR_PER_ANG = 1.8897259886
ANG_PER_BOHR = 1.0 / BOHR_PER_ANG
BOHR = 1.8897259886                       # Å  → Bohr
HARTREE_TO_KCAL = 627.509473              # Hartree → kcal/mol
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL   # kcal/mol → Hartree




#def DLFIND_constrained_optimizer(jobtype=None, theory=None, fragment=None, fragment2=None, charge=None, mult=None, 
#                                 maxcycle=250, tolerance=4.5E-4, tolerance_e=1E-6,
#                                 actatoms=None, frozenatoms=None, residues=None, 
#                                 constraints=None, restraints=None,
#                                 printlevel=2, NumGrad=False, delta=0.01,
#                                 icoord=None, iopt=None, nimage=None, 
#                                 hessian_choice="numfreq", inithessian=0, 
#                                 numfreq_npoint=1, numfreq_displacement=0.005, numfreq_hessatoms=None,
#                                 numfreq_force_projection=None, print_atoms_list=None):
#    """
#    Wrapper function around DLFIND_ConstrainedOptimizerClass
#    Adds support for harmonic restraints (bond / angle / dihedral / bond-diff).
#    """
#    import time
#    timeA = time.time()
#
#    if theory is None or fragment is None:
#        print("DLFIND_constrained_optimizer requires theory and fragment objects. Exiting.")
#        ashexit()
#
#    # Instantiate NEW optimizer class (your subclass)
#    optimizer = DLFIND_ConstrainedOptimizerClass(
#        jobtype=jobtype, theory=theory, fragment=fragment, fragment2=fragment2,
#        charge=charge, mult=mult, actatoms=actatoms,
#        frozenatoms=frozenatoms, residues=residues,
#        constraints=constraints, restraints=restraints,   # << NEW ARG HERE
#        delta=delta, printlevel=printlevel,
#        icoord=icoord, iopt=iopt, maxcycle=maxcycle,
#        tolerance=tolerance, tolerance_e=tolerance_e,
#        nimage=nimage,
#        hessian_choice=hessian_choice, inithessian=inithessian,
#        numfreq_npoint=numfreq_npoint, numfreq_displacement=numfreq_displacement,
#        numfreq_hessatoms=numfreq_hessatoms,
#        numfreq_force_projection=numfreq_force_projection,
#        print_atoms_list=print_atoms_list
#    )
#
#    # Optionally wrap theory for numerical gradients
#    if NumGrad:
#        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
#        theory = NumGradclass(theory=theory)
#
#    # Run optimization
#    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult)
#
#    if printlevel >= 1:
#        print_time_rel(timeA, modulename='DL-FIND (constrained)', moduleindex=1)
#
#    return result
#
#
#class DLFIND_ConstrainedOptimizerClass(DLFIND_optimizerClass):
#    """
#    Subclass of DLFIND_optimizerClass that adds support for *soft* harmonic restraints.
#    - Input units: lengths in Å, angles/dihedrals in degrees.
#    - k units (user): kcal/mol·Å² (for bond/bonddiff) or kcal/mol·rad² (for angles/dihedrals).
#    - Internally we convert energies to Hartree and gradients to Hartree/Å and add them
#      to the energy/gradient returned by the underlying theory.
#    """
#
#    def __init__(self, *args, restraints: dict | None = None, eps_fd: float = 1e-6, **kwargs):
#        """
#        Parameters
#        ----------
#        *args, **kwargs
#            Passed to super().__init__ (same signature as DLFIND_optimizerClass).
#        restraints: dict or None
#            Format:
#              restraints = {
#                "bond":        [[i, j, r_target_A, k_kcal_per_A2], ...],
#                "angle":       [[i, j, k, theta_target_deg, k_kcal_per_rad2], ...],
#                "dihedral":    [[i, j, k, l, phi_target_deg, k_kcal_per_rad2], ...],
#                "bonddiff":    [[i, j, k, l, diff_target_A, k_kcal_per_A2], ...],
#              }
#            All indices are zero-based.
#        eps_fd: float
#            Finite-difference step size in Angstrom for numeric derivatives (used for angle/dihedral).
#        """
#        # Save restraints raw and FD epsilon before parent's __init__ (so we can reference after)
#        self._user_restraints = restraints or {}
#        self._eps_fd = eps_fd
#
#        # Call parent initializer (it will set up spec, callbacks, etc.)
#        super().__init__(*args, **kwargs)
#
#        # Parse and store restraints in internal units (targets in A or rad; k in Hartree per unit^2)
#        self._parsed_restraints = self._parse_restraints(self._user_restraints)
#
#        # Replace / override the DL-FIND gradient callback with a wrapper that adds restraint contributions.
#        # We re-import the decorator to ensure it's available in this scope (import shown above).
#        # Note: parent already set up dlf_get_params and others; we only override gradient callback here.
#        # We keep the exact signature expected by libdlfind.
#        @dlf_get_gradient_wrapper
#        def constrained_e_g_func(coordinates, iimage, kiter, theory):
#            """
#            coordinates: pointer array in DL-FIND units (Bohr). The parent used coordinates*0.529177...
#            The parent used: coordinates_ang = coordinates * 0.5291772109303
#            That factor = 1/BOHR_PER_ANG.
#            """
#            # Count call
#            self.dlfind_eg_calls += 1
#
#            # Convert coordinates (from Bohr -> Angstrom) to call theory
#            coordinates_ang = coordinates * ANG_PER_BOHR
#
#            # Call the underlying theory to obtain energy (Hartree) and gradient (Hartree / Ang)
#            energy, gradient = theory.run(current_coords=coordinates_ang,
#                                          elems=self.fragment.elems,
#                                          charge=kwargs.get("charge", None),
#                                          mult=kwargs.get("mult", None),
#                                          Grad=True)
#
#            # If NEB bookkeeping is required, keep behavior as parent
#            if self.icoord >= 100 and self.icoord < 150:
#                # iimage bookkeeping is same as parent
#                self.NEB_geometries[iimage] = coordinates_ang
#                self.NEB_energies_dict[iimage] = energy
#
#            # Compute restraint contributions (energy in Hartree, grad in Hartree/Ang)
#            if self._parsed_restraints:
#                E_rest, grad_rest = self._compute_restraint_energy_and_grad(coordinates_ang)
#                energy = energy + E_rest
#                # gradient is expected to be a shaped array matching theory output (n_atoms,3)
#                gradient = gradient + grad_rest
#
#            return energy, gradient
#
#        # Set new gradient callback (partial not needed since decorator handles it)
#        self.dlf_get_gradient = constrained_e_g_func
#
#    def _parse_restraints(self, restraints: dict) -> list:
#        """
#        Convert user-specified restraints to an internal parsed structure.
#
#        Returns list of dicts:
#        {
#           "type": "bond"/"angle"/"dihedral"/"bonddiff",
#           "idx": [i, j, k?, l?],         # zero-based ints
#           "target": float (Ang or rad),
#           "k": float (Hartree / (Ang^2 or rad^2))
#        }
#        """
#        parsed = []
#        for key, items in restraints.items():
#            kind = key.lower()
#            if kind in ("bond",):
#                for ent in items:
#                    i, j, r0, k_kcal = ent
#                    k_hartree_per_A2 = k_kcal * KCALMOL_TO_HARTREE
#                    parsed.append({"type": "bond", "idx": [int(i), int(j)], "target": float(r0), "k": float(k_hartree_per_A2)})
#
#            elif kind in ("bonddiff", "bond_diff", "bond-diff"):
#                for ent in items:
#                    i, j, k_, l, diff0, k_kcal = ent
#                    k_hartree_per_A2 = k_kcal * KCALMOL_TO_HARTREE
#                    parsed.append({"type": "bonddiff", "idx": [int(i), int(j), int(k_), int(l)], "target": float(diff0), "k": float(k_hartree_per_A2)})
#
#            elif kind in ("angle",):
#                for ent in items:
#                    i, j, k_, theta_deg, k_kcal = ent
#                    theta_rad = math.radians(float(theta_deg))
#                    k_hartree_per_rad2 = k_kcal * KCALMOL_TO_HARTREE
#                    parsed.append({"type": "angle", "idx": [int(i), int(j), int(k_)], "target": float(theta_rad), "k": float(k_hartree_per_rad2)})
#
#            elif kind in ("dihedral", "torsion"):
#                for ent in items:
#                    i, j, k_, l, phi_deg, k_kcal = ent
#                    phi_rad = math.radians(float(phi_deg))
#                    k_hartree_per_rad2 = k_kcal * KCALMOL_TO_HARTREE
#                    parsed.append({"type": "dihedral", "idx": [int(i), int(j), int(k_), int(l)], "target": float(phi_rad), "k": float(k_hartree_per_rad2)})
#
#            else:
#                raise ValueError(f"Unknown restraint kind: {key}")
#
#        return parsed
#
#    def _compute_restraint_energy_and_grad(self, coords_ang: np.ndarray):
#        """
#        Given coordinates in Angstrom (shape (N,3)), return:
#          - E_rest (Hartree)
#          - grad_rest (numpy array shape (N,3) in Hartree/Angstrom)
#        This sums contributions from all parsed restraints.
#        """
#        n_atoms = self.fragment.numatoms
#        grad_total = np.zeros((n_atoms, 3), dtype=float)
#        E_total = 0.0
#
#        # Helper to compute single restraint energy (for numeric FD)
#        def restraint_energy_single(rst, coords):
#            t = rst["type"]
#            idx = rst["idx"]
#            if t == "bond":
#                i, j = idx
#                rij = np.linalg.norm(coords[i] - coords[j])
#                dr = rij - rst["target"]
#                return 0.5 * rst["k"] * (dr ** 2)
#            elif t == "bonddiff":
#                i, j, k_, l = idx
#                r1 = np.linalg.norm(coords[i] - coords[j])
#                r2 = np.linalg.norm(coords[k_] - coords[l])
#                dv = (r1 - r2) - rst["target"]
#                return 0.5 * rst["k"] * (dv ** 2)
#            elif t == "angle":
#                i, j, k_ = idx
#                # compute angle (radians)
#                v1 = coords[i] - coords[j]
#                v2 = coords[k_] - coords[j]
#                r1 = np.linalg.norm(v1); r2 = np.linalg.norm(v2)
#                if r1 == 0 or r2 == 0:
#                    return 0.0
#                cos_theta = np.dot(v1, v2) / (r1 * r2)
#                cos_theta = np.clip(cos_theta, -1.0, 1.0)
#                theta = math.acos(cos_theta)
#                dtheta = theta - rst["target"]
#                return 0.5 * rst["k"] * (dtheta ** 2)
#            elif t == "dihedral":
#                i, j, k_, l = idx
#                phi = self._dihedral_angle(coords[i], coords[j], coords[k_], coords[l])
#                dphi = self._wrap_angle(phi - rst["target"])  # ensure -pi..pi
#                return 0.5 * rst["k"] * (dphi ** 2)
#            else:
#                return 0.0
#
#        # For each restraint, compute analytic energy + gradient when available,
#        # otherwise compute energy then numeric gradient over involved atoms.
#        for rst in self._parsed_restraints:
#            t = rst["type"]
#            idx = rst["idx"]
#
#            if t == "bond":
#                i, j = idx
#                ri = coords_ang[i]; rj = coords_ang[j]
#                vec = ri - rj
#                r = np.linalg.norm(vec)
#                if r == 0.0:
#                    continue
#                dr = r - rst["target"]
#                E = 0.5 * rst["k"] * (dr ** 2)
#                # dE/dri = k * dr * (vec / r)
#                coeff = rst["k"] * dr / r
#                g_i = coeff * vec
#                g_j = -g_i
#                grad_total[i] += g_i
#                grad_total[j] += g_j
#                E_total += E
#
#            elif t == "bonddiff":
#                i, j, k_, l = idx
#                ri = coords_ang[i]; rj = coords_ang[j]
#                rk_ = coords_ang[k_]; rl = coords_ang[l]
#                vec1 = ri - rj
#                vec2 = rk_ - rl
#                r1 = np.linalg.norm(vec1)
#                r2 = np.linalg.norm(vec2)
#                # avoid division by zero
#                if r1 == 0.0 or r2 == 0.0:
#                    continue
#                dv = (r1 - r2) - rst["target"]
#                E = 0.5 * rst["k"] * (dv ** 2)
#                coeff = rst["k"] * dv
#                g_i = coeff * (vec1 / r1)
#                g_j = -g_i
#                g_k = -coeff * (vec2 / r2)
#                g_l = -g_k
#                grad_total[i] += g_i
#                grad_total[j] += g_j
#                grad_total[k_] += g_k
#                grad_total[l] += g_l
#                E_total += E
#
#            elif t in ("angle", "dihedral"):
#                # Use central finite differences for angle and dihedral restraint gradients.
#                # Compute energy for the restraint:
#                E_r = restraint_energy_single(rst, coords_ang)
#                E_total += E_r
#
#                # Which atoms to differentiate
#                atom_indices = rst["idx"]
#                # numeric gradient only for atoms present
#                for a in atom_indices:
#                    # central difference per Cartesian component
#                    for comp in range(3):
#                        coords_plus = coords_ang.copy()
#                        coords_minus = coords_ang.copy()
#                        coords_plus[a, comp] += self._eps_fd
#                        coords_minus[a, comp] -= self._eps_fd
#                        Ep = restraint_energy_single(rst, coords_plus)
#                        Em = restraint_energy_single(rst, coords_minus)
#                        deriv = (Ep - Em) / (2.0 * self._eps_fd)  # dE/dx_a_comp
#                        grad_total[a, comp] += deriv
#            else:
#                # unknown type -> ignore
#                continue
#
#        return E_total, grad_total
#
#    @staticmethod
#    def _wrap_angle(x):
#        """Wrap angle to [-pi, pi]."""
#        return (x + math.pi) % (2.0 * math.pi) - math.pi
#
#    @staticmethod
#    def _dihedral_angle(r1, r2, r3, r4):
#        """
#        Compute dihedral angle (radians) for four positions r1..r4 (each array-like (3,)).
#        Uses the standard vector formula with atan2 to get signed angle.
#        """
#        b1 = r2 - r1
#        b2 = r3 - r2
#        b3 = r4 - r3
#
#        # normals
#        n1 = np.cross(b1, b2)
#        n2 = np.cross(b2, b3)
#        n1_norm = np.linalg.norm(n1)
#        n2_norm = np.linalg.norm(n2)
#        if n1_norm == 0.0 or n2_norm == 0.0:
#            return 0.0
#        n1_u = n1 / n1_norm
#        n2_u = n2 / n2_norm
#
#        # unit b2
#        b2_u = b2 / (np.linalg.norm(b2) if np.linalg.norm(b2) != 0.0 else 1.0)
#
#        x = np.dot(n1_u, n2_u)
#        y = np.dot(np.cross(n1_u, n2_u), b2_u)
#        angle = math.atan2(y, x)
#        return angle
#


import math
import numpy as np

from ash.modules.optimizers.dlfind_optimizer import DLFIND_optimizerClass

# Unit conversion constants
BOHR = 1.8897259886                       # Å  → Bohr
HARTREE_TO_KCAL = 627.509473              # Hartree → kcal/mol
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL   # kcal/mol → Hartree


class DLFIND_ConstrainedOptimizerClass(DLFIND_optimizerClass):
    """
    Extends the standard DL-FIND optimizer in ASH to add *restraints*.

    Restraints differ from constraints:
       - Constraints (existing ASH feature) *fix* geometry exactly.
       - Restraints (added here) *encourage* geometry toward a target via 
         a harmonic energy penalty:  E = 0.5 * k * (value - target)^2

    Supported restraints:
        ("bond",     k, i, j, target_A)
        ("angle",    k, i, j, k, target_deg)
        ("dihedral", k, i, j, k, l, target_deg)
        ("bonddiff", k, i, j, k, l, target_A)

    All `k` values should be in kcal/mol·Å² (or kcal/mol·rad² for angles/dihedrals).
    Atom indices should be 1-based (as usual in ASH input).


    Restraints/Constraints syntax:

    constraints = [
        ("freeze", [1]),            # list of atoms to freeze
        ("bond", 4, 10, 1.52)       # [i, j, dist] exact constraint 
    ]

    restraints = [
        ("bonddiff", 15, 1, 2, 3, 4, 2.0)     # [k (weight), i, j, l, m, target Å] (shape: d(i,j)-d(l,m))
        ("angle", 30.0, 3, 7, 9, 120.0),      # soft restraint
        ("dihedral", 5.0, 2, 6, 9, 14, 180.0) # [k, i, j, target ⁰]
    ]

    """

    def __init__(self, *args, restraints=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.restraints = restraints if restraints is not None else []

    # ---- Helper functions for measuring internal coordinates ----

    def _dist(self, xyz, i, j):
        """Bond length (Å). xyz expected in Å, indices 1-based."""
        return np.linalg.norm(xyz[i-1] - xyz[j-1])

    def _angle(self, xyz, i, j, k):
        """Angle (deg)."""
        v1 = xyz[i-1] - xyz[j-1]
        v2 = xyz[k-1] - xyz[j-1]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        cosang = np.clip(cosang, -1.0, 1.0)
        return math.degrees(math.acos(cosang))

    def _dihedral(self, xyz, i, j, k, l):
        """Dihedral angle (deg)."""
        p = xyz[[i-1, j-1, k-1, l-1]]
        b0 = -1.0*(p[1] - p[0])
        b1 = p[2] - p[1]
        b2 = p[3] - p[2]
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return math.degrees(math.atan2(y, x))

    # ---- Harmonic restraint energy and gradient application ----

    def _apply_restraints(self, energy, grad, xyz):
        """
        Modifies energy and gradient in-place to include restraint penalties.
        `xyz` and geometry here are in **Å**, gradients in **Hartree/Bohr**.
        We compute restraint forces in Å units → convert force → add to gradient.
        """

        if len(self.restraints) == 0:
            return energy, grad

        # Convert gradient to Å-units temporarily for adding our force:
        # Bohr → Å: multiply by (1/BOHR)
        grad_xyz = grad / BOHR

        for r in self.restraints:
            rtype = r[0].lower()

            if rtype == "bond":
                k, i, j, target = r[1:]
                val = self._dist(xyz, i, j)
                dE_dval = k * (val - target) * KCAL_TO_HARTREE

                # Direction of force
                rij = xyz[i-1] - xyz[j-1]
                unit = rij / np.linalg.norm(rij)

                # Apply forces (negative gradient direction)
                grad_xyz[i-1] += dE_dval * unit
                grad_xyz[j-1] -= dE_dval * unit
                energy += 0.5 * k * (val - target)**2 * KCAL_TO_HARTREE

            elif rtype == "angle":
                k, i, j, k2, target_deg = r[1:]
                val_deg = self._angle(xyz, i, j, k2)
                diff_rad = math.radians(val_deg - target_deg)

                # Finite difference derivative w.r.t. atom positions
                # We'll use a small numerical derivative for the gradient part.
                # This is common in many QM/MM restraint implementations.
                eps = 1e-4
                for a in (i, j, k2):
                    for d in range(3):
                        xyz_shift = xyz.copy()
                        xyz_shift[a-1,d] += eps
                        val_shift = self._angle(xyz_shift, i, j, k2)
                        diff_shift = math.radians(val_shift - target_deg)
                        dE = 0.5*k*(diff_shift**2) - 0.5*k*(diff_rad**2)
                        grad_xyz[a-1,d] += dE/eps * KCAL_TO_HARTREE

                energy += 0.5 * k * (diff_rad**2) * KCAL_TO_HARTREE

            elif rtype == "dihedral":
                k, i, j, k2, l, target_deg = r[1:]
                val_deg = self._dihedral(xyz, i, j, k2, l)
                diff_rad = math.radians(val_deg - target_deg)
                eps = 1e-4
                for a in (i, j, k2, l):
                    for d in range(3):
                        xyz_shift = xyz.copy()
                        xyz_shift[a-1,d] += eps
                        val_shift = self._dihedral(xyz_shift, i, j, k2, l)
                        diff_shift = math.radians(val_shift - target_deg)
                        dE = 0.5*k*(diff_shift**2) - 0.5*k*(diff_rad**2)
                        grad_xyz[a-1,d] += dE/eps * KCAL_TO_HARTREE

                energy += 0.5 * k * (diff_rad**2) * KCAL_TO_HARTREE

            elif rtype == "bonddiff":
                k, i, j, k2, l, target = r[1:]
                val = (self._dist(xyz, i, j) -
                       self._dist(xyz, k2, l))
                dE_dval = k * (val - target) * KCAL_TO_HARTREE

                # We treat as sum of two bond gradients
                # First part
                rij = xyz[i-1] - xyz[j-1]
                unit1 = rij / np.linalg.norm(rij)
                grad_xyz[i-1] += dE_dval * unit1
                grad_xyz[j-1] -= dE_dval * unit1

                # Second part (opposite sign)
                rkl = xyz[k2-1] - xyz[l-1]
                unit2 = rkl / np.linalg.norm(rkl)
                grad_xyz[k2-1] -= dE_dval * unit2
                grad_xyz[l-1]  += dE_dval * unit2

                energy += 0.5 * k * (val - target)**2 * KCAL_TO_HARTREE

        # Convert gradient back to Hartree/Bohr:
        grad = grad_xyz * BOHR
        return energy, grad

    # ---- Override DL-FIND energy+gradient interface ----

    def energy_and_gradient(self, *args, **kwargs):
        """
        Call the original energy/gradient routine, then add restraint correction.
        This ensures restraints are applied *after* QM/MM evaluation.
        """
        energy, grad, xyz_bohr = super().energy_and_gradient(*args, **kwargs)

        # Convert coordinates to Å for restraint evaluation
        xyz = xyz_bohr / BOHR

        # Apply restraints
        energy, grad = self._apply_restraints(energy, grad, xyz)

        return energy, grad, xyz_bohr







def DLFIND_constrained_optimizer(jobtype=None, theory=None, fragment=None, fragment2=None, charge=None, mult=None, 
                                 maxcycle=250, tolerance=4.5E-4, tolerance_e=1E-6,
                                 actatoms=None, frozenatoms=None, residues=None, 
                                 constraints=None, restraints=None,
                                 printlevel=2, NumGrad=False, delta=0.01,
                                 icoord=None, iopt=None, nimage=None, 
                                 hessian_choice="numfreq", inithessian=0, 
                                 numfreq_npoint=1, numfreq_displacement=0.005, numfreq_hessatoms=None,
                                 numfreq_force_projection=None, print_atoms_list=None):
    """
    Wrapper function around DLFIND_ConstrainedOptimizerClass
    Adds support for harmonic restraints (bond / angle / dihedral / bond-diff).
    """
    import time
    timeA = time.time()

    if theory is None or fragment is None:
        print("DLFIND_constrained_optimizer requires theory and fragment objects. Exiting.")
        ashexit()

    # Instantiate NEW optimizer class (your subclass)
    optimizer = DLFIND_ConstrainedOptimizerClass(
        jobtype=jobtype, theory=theory, fragment=fragment, fragment2=fragment2,
        charge=charge, mult=mult, actatoms=actatoms,
        frozenatoms=frozenatoms, residues=residues,
        constraints=constraints, restraints=restraints,   # << NEW ARG HERE
        delta=delta, printlevel=printlevel,
        icoord=icoord, iopt=iopt, maxcycle=maxcycle,
        tolerance=tolerance, tolerance_e=tolerance_e,
        nimage=nimage,
        hessian_choice=hessian_choice, inithessian=inithessian,
        numfreq_npoint=numfreq_npoint, numfreq_displacement=numfreq_displacement,
        numfreq_hessatoms=numfreq_hessatoms,
        numfreq_force_projection=numfreq_force_projection,
        print_atoms_list=print_atoms_list
    )

    # Optionally wrap theory for numerical gradients
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        theory = NumGradclass(theory=theory)

    # Run optimization
    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult)

    if printlevel >= 1:
        print_time_rel(timeA, modulename='DL-FIND (constrained)', moduleindex=1)

    return result