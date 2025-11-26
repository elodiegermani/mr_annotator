from nibabel.orientations import aff2axcodes
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

# Paramètres figés
_STEP = 0.5   # échantillonnage le long des arêtes (pixels source)
_PAD  = 1     # dilatation en pixels sur la coupe cible
_OFF  = 0.0   # 0.0 = indices aux CENTRES de voxels (convention NIfTI)

def project_bbox_from_axial(ax_img, ax_affine, dst_img, dst_affine, bbox, k_ax, nx_a, ny_a, nz_a, nx_d, ny_d, nz_d):
    """
    Projette une bbox 2D d'une coupe AXIALE vers un volume cible (sagittal/coronal).
    Retour: liste triée de dicts {'slice': k_dst, 'bbox': (x1,x2,y1,y2)}.
    """
    x1, y1, w, h = map(float, bbox)
    x2 = x1 + w
    y2 = y1 + h
    x1 = round(x1 * nx_a / 512)
    x2 = round(x2 * nx_a / 512)
    y1 = round(y1 * ny_a / 512)
    y2 = round(y2 * ny_a / 512)
    #nx_a, ny_a, nz_a = ax_img.shape[:3]
    #x1, x2 = sorted((np.clip(x1, 0, nx_a-1), np.clip(x2, 0, nx_a-1)))
    #y1, y2 = sorted((np.clip(y1, 0, ny_a-1), np.clip(y2, 0, ny_a-1)))
    #k_ax   = float(np.clip(k_ax, 0, nz_a-1))

    # Échantillonnage minimal des 4 arêtes du rectangle dans l'axial
    #nx = max(2, round(np.ceil(abs(x2-x1)/_STEP))+1)
    #ny = max(2, round(np.ceil(abs(y2-y1)/_STEP))+1)
    #i = np.concatenate([
    #    np.linspace(x1+_OFF, x2+_OFF, nx),                     # bas (y=y1)
    #    np.linspace(x1+_OFF, x2+_OFF, nx),                     # haut (y=y2)
    #    np.full(ny, x1+_OFF),                                  # gauche (x=x1)
    #    np.full(ny, x2+_OFF)                                   # droite (x=x2)
    #])
    #j = np.concatenate([
    #    np.full(nx, y1+_OFF), np.full(nx, y2+_OFF),
    #    np.linspace(y1+_OFF, y2+_OFF, ny), np.linspace(y1+_OFF, y2+_OFF, ny)
    #])
    #k = np.full_like(i, k_ax+_OFF)
    #pts_src_h = np.stack([i, j, k, np.ones_like(i)], axis=1)

    # Voxel axial -> monde -> voxel cible
    p_min_ax_vox = np.array([x1, y1, nz_a - k_ax])
    p_max_ax_vox = np.array([x2, y2, nz_a - k_ax])

    p_min_ax_real = apply_affine(ax_affine, p_min_ax_vox)
    p_max_ax_real = apply_affine(ax_affine, p_max_ax_vox)

    real2vox_dst = np.linalg.inv(dst_affine).dot(ax_affine)

    p_min_dst_vox = apply_affine(real2vox_dst, p_min_ax_vox)
    p_max_dst_vox = apply_affine(real2vox_dst, p_max_ax_vox)

    out = {}
    min_k = min(round(p_min_dst_vox[-1]), round(p_max_dst_vox[-1]))
    max_k = max(round(p_min_dst_vox[-1]), round(p_max_dst_vox[-1]))
    for kz in range(min_k, max_k):
        x1 = 512 - min(p_min_dst_vox[0], p_max_dst_vox[0]) * 512 / nx_d
        x2 = 512 - max(p_min_dst_vox[0], p_max_dst_vox[0]) * 512 / nx_d
        y1 = min(p_min_dst_vox[1], p_max_dst_vox[1]) * 512 / ny_d
        y2 = max(p_min_dst_vox[1], p_max_dst_vox[1]) * 512 / ny_d
        out[kz] = (round(x1), round(x2), round(y1), round(y2))

    #xyz_h   = pts_src_h @ ax_affine.T
    #dst_ijk = xyz_h @ np.linalg.inv(dst_affine.T)  # (N,4)
    #dst_ijk = dst_ijk[:, :3]

    # Regrouper par coupe cible et calculer la bbox min entourant la projection
    # nx_d, ny_d, nz_d = dst_img.shape[:3]
    # kz = np.rround(dst_ijk[:, 2]).astype(round)
    # valid = (kz >= 0) & (kz < nz_d)
    # pts, kz = dst_ijk[valid], kz[valid]
    # out = {}
    # for kdst in np.unique(kz):
    #     p = pts[kz == kdst]
    #     x_min = 512-round(np.clip(np.floor(p[:,0].min()) - _PAD, 0, nx_d-1) * 512 / nx_d)
    #     x_max = 512-round(np.clip(np.ceil (p[:,0].max()) + _PAD, 0, nx_d-1) * 512 / nx_d)
    #     y_min = round(np.clip(np.floor(p[:,1].min()) - _PAD, 0, ny_d-1) * 512 / ny_d)
    #     y_max = round(np.clip(np.ceil (p[:,1].max()) + _PAD, 0, ny_d-1) * 512 / ny_d)
        # if x_max <= x_min and y_max >= y_min:
            # out[round(kdst)] = (x_min, x_max, y_min, y_max)
    return out