import os


class dotdict(dict):
    """dot.notation access to dictionary attributes
    
    Found at: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def prepare_paths(data_root, sample_idx, are_gt):
    fname_front = f'{sample_idx:04d}_front.png'
    fname_side = f'{sample_idx:04d}_side.png'
    
    rgb_dir = os.path.join(data_root, 'rgb/')
    rgb_path_front = os.path.join(rgb_dir, fname_front)
    rgb_path_side = os.path.join(rgb_dir, fname_side)
    
    save_dir = 'seg_gt/' if are_gt is None else 'seg/' 
    save_dirpath = os.path.join(data_root, save_dir)

    seg_path_front = os.path.join(save_dirpath, fname_front)
    seg_path_side = os.path.join(save_dirpath, fname_side)
    
    return {
        'rgb_dir': rgb_dir,
        'rgb_path_front': rgb_path_front,
        'rgb_path_side': rgb_path_side,
        'save_dir': save_dir,
        'save_dirpath': save_dirpath,
        'fname_front': fname_front,
        'fname_side': fname_side,
        'seg_path_front': seg_path_front,
        'seg_path_side': seg_path_side
    }
