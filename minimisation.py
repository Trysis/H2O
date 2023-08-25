# coding=utf-8
"""Dynamique des macromolécules.
Projet de minimisation

M1 BI-IPFB
année 2022-2023
Fait par Louiza GALOU et Roude JEAN MARIE
"""

import numpy as np
import math
import sys

# Constantes de forces
KBOND = 450  # k liaisons, kcal.mol-1.A-2
KANGLE = 55  # k valence, kcal.mol-1.rad-2

L_H2O = 0.9572  # Distance à l'équilibre
THETA_H2O = (math.pi * 104.52) / 180 # Angle à l'équilibre

FILENAME = ""  # Nom du fichier pdb

def read_coordinates(filename):
    """Renvoi sous forme de matrice (N, 3) les coordonnées d'un fichier pdb.

    Arguments
    filename : str
        fichier pdb

    Renvoi
        np.array(str), np.array(float64), np.array(str)
            header, coordonnees, tail
            Le header contient les informations des record ATOM en amont
            les coordonnées des record ATOM des atomes concernée,
            le tail contient les informations en fin des record ATOM

    """
    head_atm = []  # record atom en amont
    coordonnee = []  # coordonnee pour chaque record atom
    tail_atm = []  # record atom en aval
    with open(filename, "r") as f_in:
        for lines in f_in:
            lsp = lines.split()
            if(lsp[0] != "ATOM"): continue
            head_atm.append(lsp[:6])
            coordonnee.append([float(coord) for coord in lsp[6:9]])
            tail_atm.append(lsp[9:])

    head_atm = np.array(head_atm, dtype=object).reshape(-1, 6)
    coordonnee = np.array(coordonnee, dtype=np.float64).reshape(-1, 3)
    tail_atm = np.array(tail_atm, dtype=object).reshape(-1, 3)
    return head_atm, coordonnee, tail_atm


def E_liaison(atoms_pos):
    """Renvoi la valeur d'énergie de liaison.

    Arguments
    atoms_pos : np.array
        coordonnées atomiques de shape (N, 3)

    Renvoi
        float
            L'énergie de liaison

    """
    eliaison = 0
    # Somme des energies de liaison
    for i in atoms_pos[1:, ]:
        # atoms_pos[0, ] correspond aux coordonnées de l'Oxygène
        L = np.linalg.norm(i - atoms_pos[0, ])
        eliaison += KBOND * np.square(L - L_H2O)
    return eliaison


def E_valence(atoms_pos):
    """Renvoi la valeur d'énergie de valence.

    Arguments
    atoms_pos : np.array
        coordonnées atomiques de shape (N, 3)

    Renvoi
        float
            L'énergie de valence

    """
    pos_rela_A = atoms_pos[1, ] - atoms_pos[0, ]
    pos_rela_B = atoms_pos[2, ] - atoms_pos[0, ]
    theta = np.arccos(np.dot(pos_rela_A, pos_rela_B) / (np.linalg.norm(pos_rela_A) * np.linalg.norm(pos_rela_B)))
    return KANGLE * np.square(theta - THETA_H2O)


def E_user(atoms_pos_dict):
    """Renvoi une énergie utilisateur pour restraindre la position des atomes.

    Arguments
    atoms_pos_dict : dictionnaire
        contient les coordonnées atomiques initiales
        et contient les coordonnées atomiques actuelles

    Renvoi
        float
            Une énergie

    """
    return np.sum(np.square(atoms_pos_dict["atoms_pos"] - atoms_pos_dict["atoms_pos0"]))


def E_potentiel(atoms_pos):
    """Renvoi la valeur d'énergie potentielle.
   
    Arguments
    atoms_pos : np.array
        coordonnées atomiques de shape (N, 3)

    Renvoi
        float
            L'énergie potentielle

    """
    return E_liaison(atoms_pos) + E_valence(atoms_pos)


def E_potentiel_user(atoms_pos_dict):
    """Renvoi la valeur d'énergie potentielle avec la contrainte user.
   
    Arguments
    atoms_pos_dict : dictionnaire
        contient les coordonnées atomiques initiales
        et contient les coordonnées atomiques actuelles

    Renvoi
        float
            L'énergie potentielle

    """
    return E_liaison(atoms_pos_dict["atoms_pos"]) + E_valence(atoms_pos_dict["atoms_pos"]) + E_user(atoms_pos_dict)


def gradient_aprx(atoms_pos, delta):
    """Calcul du gradient.
   
    Arguments
    atoms_pos : np.array
        coordonnées atomiques de shape (N, 3)
    delta : float
        coefficient delta pour l'écart du coefficient directeur

    Renvoi
        np.array
            La matrice du gradient (N, 3) pour chaque composantes.

    """
    gradient = np.zeros(atoms_pos.shape, dtype=atoms_pos.dtype) # gradient (N, 3)
    deltas_xi = np.zeros(atoms_pos.shape, dtype=atoms_pos.dtype) # (N, 3) contient delta * xi à l'iteration i

    # On itère sur l'image vecteur de notre matrice (voir flat et flatten)
    for idx, atm_pos_i in enumerate(atoms_pos.flat): 
        deltas_xi.flat[idx] = delta
        gradient.flat[idx] = E_potentiel(atoms_pos + atoms_pos * deltas_xi) - E_potentiel(atoms_pos - atoms_pos * deltas_xi)
        gradient.flat[idx] = gradient.flat[idx]/(2 * atm_pos_i* deltas_xi.flat[idx])
        deltas_xi.flat[idx] = 0
    
    return gradient


def GRMS(composantes_gradient):
    """Calcul du Gradient Root mean square.

    Arguments
    composantes_gradient : np.array
        matrice du gradient de shape (N, 3)

    Renvoi
        float
            La valeur du GRMS.

    """
    return np.sqrt(np.sum(np.square(composantes_gradient))/composantes_gradient.shape[0])


def descent_grad(atoms_info, niter, seuil_grms, pas=0.01, delta=0.001, p_out=True):
    """Algorithme de steepest descent.

    Arguments
    atoms_info : np.array(str), np.array(float64), np.array(str)
        header, coordonnees, tail
    niter : int
        nombre d'itérations
    seuil_grms : float
        seuil d'arrêt du grms
    pas : float
        pas du gradient (p = -1 * pas * g)
    delta : float
        coefficient pour la dérivée numérique
    p_out : bool
        option pour permettre l'affichage des coordonnées,
        à chaque itérations

    Renvoi
        list(np.array)
            Renvoi une liste de matrice de coordonnées numpy.

    """
    heat_atm, atoms_pos, tail_atm = atoms_info
    coord_list = [np.copy(atoms_pos)] # Liste des coordonnées

    if(p_out): print_coord(heat_atm, atoms_pos, tail_atm, 0)
    # Descente du gradient
    for i in range(niter):
        gradient = gradient_aprx(atoms_pos, delta)
        grms_i = GRMS(gradient.flatten())
        if grms_i < seuil_grms:
            break
        gradient_oppose = -1 * gradient
        atoms_pos = atoms_pos + pas * gradient_oppose
        if(p_out): print_coord(heat_atm, atoms_pos, tail_atm, i + 1)
        coord_list.append(np.copy(atoms_pos))
        
    return coord_list

def atoms_RMSF(coord_list):
    """Renvoi les valeurs de RMSF pour chaque atomes.

    Arguments
    coord_list : liste de coordonnées à chaque itérations

    Renvoi
        list
            Les valeur de RMSF
    """
    rmsf_list = []
    for atm_i in range(coord_list[0].shape[0]):
        atm_pos = np.array([atm[atm_i, ] for atm in coord_list])
        pos_mean = atm_pos.mean(axis = 0).reshape(-1, 3)
        t = atm_pos.shape[0]

        rmsf_list.append(np.sqrt(np.sum(np.square(atm_pos - pos_mean))/t))

    return rmsf_list

def atom_record_str(hinfo, atm, tinfo):
    """Renvoi dans le bon format PDB pour un record ATOM.

    Arguments
    hinfo : np.array(str)
        header d'un record ATOM
    atm : np.array(float64)
        coordonnees d'un record ATOM
    tinfo : np.array(str)
        tail d'un record ATOM

    Renvoi
        str
            une ligne record ATOM.

    """
    atm_record = "" # Contiendra une section ATOM complète
    atm_record += f"{hinfo[0]:6s}{int(hinfo[1]):5d} {hinfo[2][0]:^4s}{hinfo[3]:3s} {hinfo[4]:1s}"
    atm_record += f"{1:4d}{'':1s}   {atm[0]:8.3f}{atm[1]:8.3f}{atm[2]:8.3f}{float(tinfo[0]):6.2f}"
    atm_record += f"{float(tinfo[1]):>6.0f}          {tinfo[2]:>2s}{'':2s}"

    return atm_record


def print_coord(head_atm, atoms_pos, tail_atm, num=-1):
    """Affiche la section MODEL/ENDML avec les coordonnées atomiques.

        Permet de générer dans le bon format pdb, l'affichage des
        coordonnées atomiques de chacun des atomes à l'aide des
        informations du head, tail et des coordonnées.

    Arguments
    head_atm : np.array(str)
        header de plusieurs record ATOM
    atoms_pos : np.array(float64)
        coordonnees de plusieurs record ATOM
    tail_atm : np.array(str)
        tail de plusieurs record ATOM
    num : int
        numéro du modèle actuel

    """
    if(num>-1):print(f"MODEL {num}")
    for hinfo, atm, tinfo in zip(head_atm, atoms_pos, tail_atm):
        print(atom_record_str(hinfo, atm, tinfo))
    print(f"CONECT{1:>5d}{2:>5d}{3:>5d}")
    if(num>-1):print(f"ENDMDL {num}")


def print_Epotentiel(atoms_pos, temps=-1):
    """Affiche la valeurs d'énergie potentielles.

    Arguments
    atoms_pos : np.array
        coordonnées atomiques de shape(N, 3)
    temps : int
        itération à laquelle l'énergie a été calculé.

    """
    if(temps<=-1):
        print(E_potentiel(atoms_pos))
    print(f"{temps} {E_potentiel(atoms_pos)}")


def print_Epotentiel_user(atoms_pos_dict, temps=-1):
    """Affiche la valeurs d'énergie potentielles avec e_user.

    Arguments
    atoms_pos_dict : dictionnaire
        contient les coordonnées atomiques initiales
        et contient les coordonnées atomiques actuelles
    temps : int
        itération à laquelle l'énergie a été calculé.

    """
    if(temps<=-1):
        print(E_potentiel_user(atoms_pos_dict))
    print(f"{temps} {E_potentiel_user(atoms_pos_dict)}")


if __name__ == "__main__":
    if len(sys.argv) < 2 | len(sys.argv) > 6:
        sys.exit("python minimisation.py file.pdb")

    FILENAME = sys.argv[1]
    atoms_info = read_coordinates(FILENAME)
    niter = 500 # nombre d'itérations
    seuil = 0.01 # seuil grms
    pas = 0.0001 # pas pour le gradient
    delta = 0.00001 # delta pour la composante_i * delta

    if len(sys.argv) >= 3:
        niter = int(sys.argv[2])
    if len(sys.argv) >= 4:
        seuil = float(sys.argv[3])
    if len(sys.argv) >= 5:
        pas = float(sys.argv[4])
    if len(sys.argv) >= 6:
        delta = float(sys.argv[5])

    # Descente du gradient
    coordos = descent_grad(atoms_info, niter, seuil, pas, delta)
    rmsf_list = atoms_RMSF(coordos)