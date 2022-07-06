import os
import torch
import smplx

SMPL_NUM_KPTS = 23
SMPLX_NUM_KPTS = 21

MODEL_FILES_DIR = './model_files/'


def set_shape(model, shape_coefs):
    # TODO: Implement STAR support.
    #if type(model) == smplx.star.STAR:
    #    return model(pose=torch.zeros((1, 72), device='cpu'), betas=shape_coefs, trans=torch.zeros((1, 3), device='cpu'))
    #else:
    shape_coefs = torch.tensor(shape_coefs, dtype=torch.float32)
    return model(betas=shape_coefs, return_verts=True)


def create_model(gender, num_coefs=10, model_type='smpl'):
    # TODO: Implement STAR support.
    if model_type == 'star':
        return smplx.star.STAR()
    else:
        body_pose = torch.zeros((1, SMPL_NUM_KPTS * 3))
        return smplx.create(MODEL_FILES_DIR, model_type=model_type,
                            gender=gender, use_face_contour=False,
                            num_betas=num_coefs,
                            body_pose=body_pose,
                            ext='npz')
