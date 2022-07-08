import copy
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import ase
from ase.io import read, write
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import stack, rotate
import numpy as np
import statistics
import itertools
from itertools import permutations
import math
import os
from pathlib import Path
import shutil
import pandas as pd


def crystal_lattices137():
    a, c = 3.56559516, 5.16231468
    s137 = Structure.from_spacegroup(
        sg=137,
        lattice=Lattice.tetragonal(a, c),
        species=['Hf', 'O'],
        coords=[[0.0, 0.0, 0.5], [0.0, 0.5, 0.20053958]],
    )
    s137 = AseAtomsAdaptor.get_atoms(s137)
    s137 *= [2, 1, 1]
    rotate(s137, [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0])
    s137.set_cell([a * 2 ** 0.5, a * 2 ** 0.5, c])
    s137.wrap()
    rotate(s137, [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    s137.set_cell([c, a * 2 ** 0.5, a * 2 ** 0.5])
    return s137


def crystal_lattices29():
    s29 = Structure.from_spacegroup(
        sg=29,
        lattice=Lattice.orthorhombic(5.18210103, 4.97802758, 4.99815551),
        species=['Hf', 'O', 'O'],
        coords=[[4.67124112e-01, 2.31967324e-01, 7.47923199e-01], [1.31131890e-01, 4.26941569e-01, 6.11986110e-01],
                [2.64477538e-01, 9.58631802e-01, 5.00252042e-01]],
    )

    s29 = AseAtomsAdaptor.get_atoms(s29)
    s29.translate([-s29.positions[0, 0], -s29.positions[2, 1], -s29.positions[0, 2]])
    s29.wrap()
    return s29


def create_slab(s137averaged, s29averaged, size_s137, size_s29, slab_orientation):
    slab_size_s137, slab_size_s29 = np.ones(3, dtype=int), np.ones(3, dtype=int)
    slab_size_s137[slab_orientation], slab_size_s29[slab_orientation] = size_s137, size_s29
    #s137_copy, s29_copy = copy.deepcopy(s137averaged), s29averaged
    #s29_copy.set_cell(s29_copy.cell.cellpar()[:3])  # sieht so aus als müssten die zellen nur werte auf de rdiagonalen zellmatrix haben sonst 0
    s137_copy = s137averaged*slab_size_s137
    s29_copy = s29averaged* slab_size_s29            #sieht so aus als müssten die zellen nur werte auf de rdiagonalen zellmatrix haben sonst 0
    si = stack( s137_copy, s29_copy, slab_orientation)         # wenn nicht wird der slab nicht als atoms objekt erkannt
    return si


def create_minislab(crys1, crys2, slab_orientation):
    si = stack(crys1, crys2, int(slab_orientation))
    return si


def Matrixerstellung(j):
    alphaxj = j[0]
    alphayj = j[1]
    alphazj = j[2]
    Rgesamt = np.array([[[1, 0, 0],
                         [0, math.cos(alphaxj), -math.sin(alphaxj)],
                         [0, math.sin(alphaxj), math.cos(alphaxj)]],
                        [[math.cos(alphayj), 0, math.sin(alphayj)],
                         [0, 1, 0],
                         [-math.sin(alphayj), 0, math.cos(alphayj)]],
                        [[math.cos(alphazj), -math.sin(alphazj), 0],
                         [math.sin(alphazj), math.cos(alphazj), 0],
                         [0, 0, 1]]], )
    return Rgesamt


def drehung():
    l = list(permutations(range(0, 3)))
    alphas = list(itertools.product([0, 1 * math.pi / 2, 2 * math.pi / 2, 3 * math.pi / 2], repeat=3))
    Drehmatrizen = np.zeros([3, 3])
    Drehmatrizen = Drehmatrizen[np.newaxis, ...]
    for j in alphas:
        Rgesamt = np.rint(Matrixerstellung(j))
        for i in l:
            Rotation = np.rint(Rgesamt[i[0]].dot(Rgesamt[i[1]]).dot(Rgesamt[i[2]]))
            Rotation = Rotation[np.newaxis, ...]
            if np.any(np.all(Drehmatrizen == Rotation, axis=(1, 2))) == False:
                Drehmatrizen = np.vstack([Drehmatrizen, Rotation])
    Drehmatrizen = np.delete(Drehmatrizen, 0, axis=0)
    return Drehmatrizen


def spiegelung(spiegelungsart, crystal):
    spiegelungsarten = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, 1], [1, -1, -1], [-1, -1, -1]])
    multiplikator = [spiegelungsarten[
                         spiegelungsart]]  # Erzeugung eines Multiplikators zur transformation der Positionsmatrix der Atome
    # multiplikator = np.repeat(multiplikator,len(crystal.get_positions()),axis=0)
    # crystal.set_positions(crystal.get_positions()*multiplikator)
    # crystal.wrap()
    return multiplikator


# return crystal

def Matrixoperation(Drehmatrizen):
    spiegelungsarten = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                 [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                                 [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                 [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                 [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]])
    Vergleichsmatrix = np.zeros([3, 3])
    Vergleichsmatrix = Vergleichsmatrix[np.newaxis, ...]
    for i in spiegelungsarten:
        for j in Drehmatrizen:
            FertigeMatrix = np.rint(j.dot(i))[np.newaxis, ...]
            if np.any(np.all(Vergleichsmatrix == FertigeMatrix, axis=(1, 2))) == False:
                Vergleichsmatrix = np.vstack([Vergleichsmatrix, FertigeMatrix])
    Vergleichsmatrix = np.delete(Vergleichsmatrix, 0, axis=0)
    return Vergleichsmatrix


"""
def abstandskontrolle(s137s29, distance):  # erstmal für einzelne Kristalle

    indexarray = [i for i in range(len(s137s29.get_atomic_numbers()))
                  if s137s29.get_atomic_numbers()[i] < 40]
    distances137s29 = s137s29.get_distances(indexarray.pop(0), indexarray)
    #ic(s137s29.cell.cellpar()[:3])
    #ic(np.max(s137s29.cell.cellpar()[:3]))
    celllength = np.max(s137s29.cell.cellpar()[:3])/len(s137s29.get_atomic_numbers())*12
    ic(celllength)
    #ic(len(s137s29.get_atomic_numbers())/12)
    distances137s29 = (distances137s29%celllength)/distance #modulu von einzel Zelllänge
    #print(distances137s29)
    ic(distances137s29)
    return any(distances137s29 > 1.14)
"""

def abstandskontrolle(s137s29,distance):	    #erstmal für einzelne Kristalle
    indexarray = [i for i in range(len(s137s29.get_atomic_numbers()))
                  if s137s29.get_atomic_numbers()[i] > 40]
    distances137s29 = s137s29.get_distances(indexarray.pop(0), indexarray)/distance
    #ic(distances137s29)
    return any(distances137s29 < 0.7)

def obKristallschonerstellt_s137(s137, VergleichsmatrixKristall_s137):
    anzahlatome = len(s137.get_positions())
    for _ in VergleichsmatrixKristall_s137:
        counter = 0
        for i in s137.get_positions():
            for j in _:
                if all(np.round(i,2) == np.round(j,2)):
                    counter += 1
        if counter == anzahlatome:
            return False, VergleichsmatrixKristall_s137

    VergleichsmatrixKristall_s137 = np.append(VergleichsmatrixKristall_s137, s137.get_positions())
    VergleichsmatrixKristall_s137 = VergleichsmatrixKristall_s137.reshape(int(len(VergleichsmatrixKristall_s137) / (anzahlatome * 3)),
                                                                anzahlatome, 3)
    return True, VergleichsmatrixKristall_s137

def obKristallschonerstellt_s29(s29, VergleichsmatrixKristall_s29):
    anzahlatome = len(s29.get_positions())
    for _ in VergleichsmatrixKristall_s29:
        counter = 0
        for i in s29.get_positions():
            for j in _:
                if all(np.round(i,2) == np.round(j,2)):
                    counter += 1
        if counter == anzahlatome:
            return False, VergleichsmatrixKristall_s29

    VergleichsmatrixKristall_s29 = np.append(VergleichsmatrixKristall_s29, s29.get_positions())
    VergleichsmatrixKristall_s29 = VergleichsmatrixKristall_s29.reshape(int(len(VergleichsmatrixKristall_s29) / (anzahlatome * 3)),
                                                                anzahlatome, 3)
    return True, VergleichsmatrixKristall_s29

def ifTetraOrientationisSlabOrientation(s137_copy, s29_copy, slab_orientation):
    #_s137_copy = s137
    #_s29_copy = s29
    cells137 = s137_copy.get_cell_lengths_and_angles()[:3]
    cells29 = s29_copy.get_cell_lengths_and_angles()[:3]
    cell_mean = np.mean([cells137, cells29], axis=0)
    #ic(cell_mean,cells137,cells29)
    TETR_ORIENTATION = np.argmax(np.diag(s137_copy.get_cell()))

    SLAB_ORIENTATION = slab_orientation
    if TETR_ORIENTATION == SLAB_ORIENTATION:
        mask = np.full(3, True)
        mask[SLAB_ORIENTATION] = False
        cell_mean[mask] = np.mean(cell_mean[mask])
        new_cells29 = cell_mean.copy()
        new_cells29[SLAB_ORIENTATION] = cells29[SLAB_ORIENTATION]
        new_cells137 = cell_mean.copy()
        new_cells137[SLAB_ORIENTATION] = cells137[SLAB_ORIENTATION]
    else:
        new_cells29 = cell_mean.copy()
        new_cells29[SLAB_ORIENTATION] = cells29[SLAB_ORIENTATION]
        new_cells137 = cell_mean.copy()
        mask = np.full(3, True)
        mask[SLAB_ORIENTATION] = False
        mask[TETR_ORIENTATION] = False
        new_cells137[SLAB_ORIENTATION] = cell_mean[np.arange(3)[mask]]

    #_s137_copy.set_cell(Lattice.orthorhombic(*new_cells137).matrix, scale_atoms=True)
    s137_copy.set_cell(new_cells137)
    s137_copy.wrap(pretty_translation=True)
    #_s29_copy.set_cell(Lattice.orthorhombic(*new_cells29).matrix, scale_atoms=True)
    s29_copy.set_cell(new_cells29)
    s29_copy.wrap()#pretty_translation=True)#center=[0.5,0.5,0.5],eps=0.2)#pbc= [True,False,False])#eps=0.1)#center=[0.5,0.5,0.5])#pretty_translation=True)
    return s137_copy, s29_copy

def interface_check(s137s29, s137averaged,size_s137,):
    #index = len(s137averaged.get_atomic_numbers())-1
    #oxygens = s137s29[s137s29.get_atomic_numbers() ==8]
    X = s137s29.get_all_distances().flatten().tolist()
    #X = oxygens.get_all_distances().flatten().tolist()
    #print(X)
    Y=[k for k in X if k > 0.01]
    print(any(np.array(Y) <0.5))
    #print()

    #view(oxygens)

    if any(np.array(Y) <0.5):
        print("false interface")
        return False
    else:
        print("interphase okay")
        return True
    """if s137s29.get_atomic_numbers()[index] > 40 and s137s29.get_atomic_numbers()[index+1] < 40:
        return True
    if s137s29.get_atomic_numbers()[index] < 40 and s137s29.get_atomic_numbers()[index + 1] > 40:
        return True
    else:
        return False
    """

def storagefunction(new_dir, s137, s29, s137s29):
    new_dir.mkdir(parents=True, exist_ok=True)
    ase.io.write(Path(new_dir, "slab"), s137s29, format="cif")
    ase.io.write(Path(new_dir, "s137"), s137, format="cif")
    ase.io.write(Path(new_dir, "s29"), s29, format="cif")
    shutil.copy("relax.py", new_dir)
    # shutil.copy("run", new_dir)
    shutil.copy("/home/richard/mountHPD/ka39zis/aims/run_files/run_aims_cm2_tiny_4nodes", new_dir)
    shutil.copy("/home/richard/mountHPD/ka39zis/aims/run_files/run_aims_hm_mech_4nodes", new_dir)

def create_all_s137(matrizen):
    identifier_s137=[]
    s137 = crystal_lattices137()
    view(s137)
    print(s137.get_cell())
    oxygens = s137[s137.get_atomic_numbers() == 8]
    vergleichsmatrix_s137 = oxygens.get_positions()[np.newaxis, ...]
    #vergleichsmatrix_s137 = s137.get_positions()[np.newaxis, ...]
    s137_list=[]
    X=[i for i in matrizen if i[0][0]!=-1]
    #print(len(X),X)
    for i in X:
        s137 = crystal_lattices137()
        s137.set_scaled_positions(np.squeeze(s137.get_scaled_positions().dot(i)[np.newaxis, ...], axis=0))
        s137.wrap()
        s137.set_cell(np.diag(abs(s137.cell.cellpar()[:3].dot(i))))
        s137.wrap()
        s137_copy = copy.deepcopy(s137)
        oxygens = s137[s137.get_atomic_numbers() ==8]
        neu,vergleichsmatrix_s137 = obKristallschonerstellt_s137(oxygens, vergleichsmatrix_s137)
        #print("s137",neu)
        if neu:
            order = [1, 2, 3]
            #view(s137_copy)
            s137_list.append(s137)
            new_order = i.dot(order).tolist()

            for n,i in enumerate(new_order):
                if i ==1.0 or i == -1.0:
                    new_order[n]="a"
                if i ==-1.0:
                    new_order[n] = "-a"
                if i == 2.0:
                    new_order[n] = "b"
                if i ==-2.0:
                    new_order[n] = "-b"
                if i == 3.0:
                    new_order[n] = "b"
                if i ==-3.0:
                    new_order[n] = "-b"
            print(new_order)
            identifier_s137.append(new_order)
            #view(s137)
            """#half_s137 =s137[:6]    Idee den half_shift hinzuzufügen
            #half_s137.wrap(pretty_translation=True)
            #s137.wrap()
            #view(s137)

            #view(s137[6:-6])
            #neu, vergleichsmatrix_s137 = obKristallschonerstellt_s137(s137, vergleichsmatrix_s137)
            #if neu:
            #    s137_list.append(s137)"""
    return s137_list, identifier_s137

def create_all_s29(matrizen):
    identifier_s29=[]
    s29_list = []
    s29 = crystal_lattices29()
    vergleichsmatrix_s29 = s29.get_positions()[np.newaxis, ...]
    X = [i for i in matrizen if i[0][0] != -1]
    for i in X:
        s29 = crystal_lattices29()
        s29.set_scaled_positions(np.squeeze(s29.get_scaled_positions().dot(i)[np.newaxis, ...], axis=0))
        s29.wrap()
        s29.set_cell(np.diag(abs(s29.cell.cellpar()[:3].dot(i))))
        s29.wrap()
        s29_copy = copy.deepcopy(s29)
        neu, vergleichsmatrix_s29 = obKristallschonerstellt_s29(s29, vergleichsmatrix_s29)
        #print("s29",neu)
        if neu:
            #view(s29_copy)
            order = [1, 2, 3]
            s29_list.append(s29_copy)
            new_order = i.dot(order).tolist()
            for n, i in enumerate(new_order):
                if i == 1.0 or i == -1.0:
                    new_order[n] = "a"
                if i == -1.0:
                    new_order[n] = "-a"
                if i == 2.0:
                    new_order[n] = "b"
                if i == -2.0:
                    new_order[n] = "-b"
                if i == 3.0:
                    new_order[n] = "c"
                if i == -3.0:
                    new_order[n] = "-c"
            identifier_s29.append(new_order)
            print(new_order)

    return s29_list, identifier_s29

if __name__ == "__main__":
    slab_orientations = [0, 1, 2]
    size_s137, size_s29 = 1, 5
    k = 0
    x = "s137"
    y = "s29"
    z = x + y
    s137 = crystal_lattices137()
    #s29 = crystal_lattices29()
    #view(s29)
    #view(s29)
    indexarray = [i for i in range(len(s137.get_atomic_numbers()))
                  if s137.get_atomic_numbers()[i] > 40]
    distance = statistics.mean(s137.get_distances(indexarray.pop(0), indexarray))
    """indexarray = [i for i in range(len(s137.get_atomic_numbers()))
                  if s137.get_atomic_numbers()[i] < 40]

    distance = np.max(s137.get_distances(indexarray[0], indexarray))"""
    #view(s137)
    Drehmatrizen = np.array(drehung())
    matrizen = Matrixoperation(Drehmatrizen)
    allekristalle = []
    si = []
    s137array = []
    #VergleichsmatrixKristall_s137 = s137.get_positions()[np.newaxis, ...]
    #VergleichsmatrixKristall_s29 = s29.get_positions()[np.newaxis, ...]
    frame=[]
    s137_list, identifier_s137=create_all_s137(matrizen)
    s29_list, identifier_s29=create_all_s29(matrizen)
    #k=0
    #print(len(s29_list),len(s137_list))
    #print(matrizen)


    for index_s137,j in enumerate(s137_list):
        for index_s29, i in enumerate(s29_list):
            """#view(j)
            #s137averaged, s29averaged = ifTetraOrientationisSlabOrientation(j, i, 0)
            #view(s137averaged)
            #if_tetra_is_slab(j,i,0)
            #view(j)
            #view(s137averaged)
        
        
            #s137s29 = create_slab(j,i, size_s137, size_s29,0)
            #s137s29 = create_minislab(s137averaged, s29averaged,0)# size_s137, size_s29, 0)
            view(s137s29)
        
            #s137s29 = create_slab(j, i, size_s137, size_s29, 0)
            #view(s137s29)
            #os.mkdir(r"/home/richard/scripts/s137s29/".join(str(np.array([1, 2, 3]).dot(i))))
            #new_dir_name = str(str(np.array([1, 2, 3]).dot(i)) + str(round(j)))
            new_dir = Path('/home/richard/mountHPD/ka39zis/aims/interphases/1_5_slabs/' + str(k))
            #new_dir = r"/home/richard/mountHPD/ka39zis/aims/interphases/1_5_slabs/" + str(k)
            storagefunction(new_dir, j, i, s137s29)
            break"""
            #k += 1"""
            try:
                #s137s29 = create_minislab(s137averaged, s29averaged,0)# size_s137, size_s29, 0)


                s137s29 = create_slab(j, i, 1, 5, 0)
                #view(s137s29)
                #print(interface_check(s137s29,s137averaged,0))
                if interface_check(s137s29,j,0):
                    new_dir = Path(r"/home/richard/mountHPD/ka39zis/aims/interphases/test/"+str(k))
                    storagefunction(new_dir, j, i, s137s29)
                    frame.append([int(k), identifier_s137[index_s137], identifier_s29[index_s29]])
                    k+=1
                    #view(s137s29)
                    print(k)
                    #print("stored")
            except:
                print("nicht geklappt")

    dataframe = pd.DataFrame(frame,columns=["Folder","ident_s137","ident_s29"])
    dataframe.to_csv(r"/home/richard/mountHPD/ka39zis/aims/interphases/test/identity.csv",index=False)
    view(s137s29)
        #view(i)
        #view(j)

    """
    for i in Matrizen:
        s137 = crystal_lattices137()
        s137.set_scaled_positions(np.squeeze(s137.get_scaled_positions().dot(i)[np.newaxis, ...], axis=0))
        s137.wrap()
        s137.set_cell(np.diag(s137.cell.cellpar()[:3].dot(i)))
        s137.wrap()
        s137_copy = copy.deepcopy(s137)
        schonvorhanden, VergleichsmatrixKristall = obKristallschonerstellt(s137, VergleichsmatrixKristall)
        if not schonvorhanden:
            #for j in slab_orientations:
                s29 = crystal_lattices29()
                s137averaged, s29averaged = ifTetraOrientationisSlabOrientation(s137_copy, s29, j)
                #view(s137averaged)
                #view(s137averaged)
                #view(s29averaged)
                #s137s29 = create_minislab(s137averaged, s29, j)
                #try:
                s137s29 = create_slab(s137averaged, s29averaged, size_s137, size_s29, j)
                view(s137s29)
                print(abstandskontrolle(s137s29,distance))
                if not abstandskontrolle(s137s29, distance):
                    si.append(s137s29)
                    # s137array.append(s137averaged)
                    # print("hier")
                    view(s137s29)
                    # print(sl.get_cell())
                    # print(np.argmax(sl.get_cell()))
                    # os.mkdir(r"/home/richard/scripts/s137s29/".join(str(np.array([1,2,3]).dot(i))))
                    new_dir_name = str(str(np.array([1, 2, 3]).dot(i)) + str(round(j)))
                    # view(s137s29)
                    new_dir = Path('/home/richard/mountHPD/ka39zis/aims/interphases/Minislabs1_3/' + new_dir_name)
                    storagefunction(new_dir, s137averaged, s29, s137s29)
    """
    """
            except:
                    print("continue")
                    continue


                
                s137s29 = create_slab(s137averaged, s29averaged, size_s137, size_s29, j)
                view(s137s29)
                #ic(interface_check(s137s29, s137averaged, size_s137))
                #if not abstandskontrolle(s137s29, distance):
                #if interface_check(s137s29, s137averaged, size_s137):
                #si.append(s137s29)
                # print(sl.get_cell())
                # print(np.argmax(sl.get_cell()))
                # os.mkdir(r"/home/richard/scripts/s137s29/".join(str(np.array([1,2,3]).dot(i))))
                new_dir_name = str(str(np.array([1, 2, 3]).dot(i)) + str(round(j)))
                new_dir = Path('/home/richard/mountHPD/ka39zis/aims/interphases/Minislabs1_3/' + new_dir_name)
                storagefunction(new_dir, s137averaged, s29averaged, s137s29)
"""
