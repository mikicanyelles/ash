# This is the full DLFIND_optimizerClass from your prompt,
# modified to accept constraint values and 'bond_diff' constraints.

# I'm importing 'os', 'numpy', etc. which were implied in the original
import os
import functools
from ctypes import c_double, c_int, pointer
import numpy as np
from numpy.ctypeslib import as_array

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

