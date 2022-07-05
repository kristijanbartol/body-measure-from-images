from typing import List
import numpy as np
import trimesh

from model import create_model, set_shape
    
    
class MeasurementTypes():
        
    CIRCUMFERENCE = 'circumference'
    LENGTH = 'length'       # maps to distance
    BREADTH = 'breadth'     # maps to distance
    DEPTH = 'depth'         # maps to distance
    SPAN = 'span'           # maps to distance
    DISTANCE = 'distance'
    HEIGHT = 'height'

    
class MeshIndices():
    
    # Mesh landmark indexes.
    HEAD_TOP = 412
    LEFT_HEEL = 3463
    LEFT_NIPPLE = 598
    BELLY_BUTTON = 3500
    INSEAM_POINT = 3149
    LEFT_SHOULDER = 3011
    RIGHT_SHOULDER = 6470
    LEFT_CHEST = 1423
    RIGHT_CHEST = 4896
    LEFT_WAIST = 631
    RIGHT_WAIST = 4424
    UPPER_BELLY_POINT = 3504
    REVERSE_BELLY_POINT = 3502
    LEFT_HIP = 1229
    RIGHT_HIP = 4949
    LEFT_MID_FINGER = 2445
    RIGHT_MID_FINGER = 5906
    LEFT_WRIST = 2241
    RIGHT_WRIST = 5702
    LEFT_INNER_ELBOW = 1663
    RIGHT_INNER_ELBOW = 5121

    SHOULDER_TOP = 3068
    LOW_LEFT_HIP = 3134
    LEFT_ANKLE = 3334

    LOWER_BELLY_POINT = 1769
    FOREHEAD_POINT = 336
    NECK_POINT = 3049
    HIP_POINT = 1806
    RIGHT_BICEP_POINT = 6281
    RIGHT_FOREARM_POINT = 5084
    RIGHT_THIGH_POINT = 4971
    RIGHT_CALF_POINT = 4589
    RIGHT_ANKLE_POINT = 6723

    # Mesh measurement idnexes.
    OVERALL_HEIGHT = (HEAD_TOP, LEFT_HEEL)
    SHOULDER_TO_CROTCH_HEIGHT = (SHOULDER_TOP, INSEAM_POINT)
    NIPPLE_HEIGHT = (LEFT_NIPPLE, LEFT_HEEL)
    NAVEL_HEIGHT = (BELLY_BUTTON, LEFT_HEEL)
    INSEAM_HEIGHT = (INSEAM_POINT, LEFT_HEEL)

    SHOULDER_BREADTH = (LEFT_SHOULDER, RIGHT_SHOULDER)
    CHEST_WIDTH = (LEFT_CHEST, RIGHT_CHEST)
    WAIST_WIDTH = (LEFT_WAIST, RIGHT_WAIST)
    TORSO_DEPTH = (UPPER_BELLY_POINT, REVERSE_BELLY_POINT)
    HIP_WIDTH = (LEFT_HIP, RIGHT_HIP)

    ARM_SPAN_FINGERS = (LEFT_MID_FINGER, RIGHT_MID_FINGER)
    ARM_SPAN_WRIST = (LEFT_WRIST, RIGHT_WRIST)
    ARM_LENGTH = (LEFT_SHOULDER, LEFT_WRIST)
    FOREARM_LENGTH = (LEFT_INNER_ELBOW, LEFT_WRIST)
    INSIDE_LEG_LENGTH = (LOW_LEFT_HIP, LEFT_ANKLE)
    
    # Circumference indices.
    WAIST_INDICES = [3500, 1336, 917, 916, 919, 918, 665, 662, 657, 654, 631, 632, 720, 799, 796, 890, 889, 3124, 3018, \
            3019, 3502, 6473, 6474, 6545, 4376, 4375, 4284, 4285, 4208, 4120, 4121, 4142, 4143, 4150, 4151, 4406, 4405, \
            4403, 4402, 4812]

    CHEST_INDICES = [3076, 2870, 1254, 1255, 1349, 1351, 3033, 3030, 3037, 3034, 3039, 611, 2868, 2864, 2866, 1760, 1419, 741, \
            738, 759, 2957, 2907, 1435, 1436, 1437, 1252, 1235, 749, 752, 3015, 4238, 4237, 4718, 4735, 4736, 4909, ]

    HIP_INDICES = [1806, 1805, 1804, 1803, 1802, 1801, 1800, 1798, 1797, 1796, 1794, 1791, 1788, 1787, 3101, 3114, 3121, \
            3098, 3099, 3159, 6522, 6523, 6542, 6537, 6525, 5252, 5251, 5255, 5256, 5258, 5260, 5261, 5264, 5263, 5266, \
            5265, 5268, 5267]

    THIGH_INDICES = [877, 874, 873, 848, 849, 902, 851, 852, 897, 900, 933, 936, 1359, 963, 908, 911, 1366]

    CALF_INDICES = [1154, 1372, 1074, 1077, 1470, 1094, 1095, 1473, 1465, 1466, 1108, 1111, 1530, 1089, 1086]

    ANKLE_INDICES = [3322, 3323, 3190, 3188, 3185, 3206, 3182, 3183, 3194, 3195, 3196, 3176, 3177, 3193, 3319]

    WRIST_INDICES = [1922, 1970, 1969, 2244, 1945, 1943, 1979, 1938, 1935, 2286, 2243, 2242, 1930, 1927, 1926, 1924]

    ELBOW_INDICES = [1573, 1915, 1914, 1577, 1576, 1912, 1911, 1624, 1625, 1917, 1611, 1610, 1607, 1608, 1916, 1574]

    BICEP_INDICES = [789, 1311, 1315, 1379, 1378, 1394, 1393, 1389, 1388, 1233, 1232, 1385, 1381, 1382, 1397, 1396, 628, 627]

    NECK_INDICES = [3068, 1331, 215, 216, 440, 441, 452, 218, 219, 222, 425, 426, 453, 829, 3944, 3921, 3920, 3734, \
            3731, 3730, 3943, 3935, 3934, 3728, 3729, 4807]

    HEAD_INDICES = [336, 232, 235, 1, 0, 3, 7, 136, 160, 161, 166, 167, 269, 179, 182, 252, 253, 384, 3765, 3766, 3694, \
            3693, 3782, 3681, 3678, 3671, 3672, 3648, 3518, 3513, 3514, 3515, 3745, 3744]


class MeshMeasurements():
    
    LABELS_TO_NAMES = {
        'A': 'head',
        'B': 'neck',
        'C': 'shoulder_to_crotch',
        'D': 'chest',
        'E': 'waist',
        'F': 'hip',
        'G': 'wrist',
        'H': 'bicep',
        'I': 'forearm',
        'J': 'arm',
        'K': 'inside_leg',
        'L': 'thigh',
        'M': 'calf',
        'N': 'ankle',
        'O': 'shoulder'
    }
    
    NAMES_TO_LABELS = {
        'head': 'A',
        'neck': 'B',
        'shoulder_to_crotch': 'C',
        'chest': 'D',
        'waist': 'E',
        'hip': 'F',
        'wrist': 'G',
        'bicep': 'H',
        'forearm': 'I',
        'arm': 'J',
        'inside_leg': 'K',
        'thigh': 'L',
        'calf': 'M',
        'ankle': 'N',
        'shoulder': 'O',
        'overall': 'N'
    }  
    
    _DEFAULT_TYPES = {
        'A': MeasurementTypes.CIRCUMFERENCE,
        'B': MeasurementTypes.CIRCUMFERENCE,
        'C': MeasurementTypes.HEIGHT,
        'D': MeasurementTypes.CIRCUMFERENCE,
        'E': MeasurementTypes.CIRCUMFERENCE,
        'F': MeasurementTypes.CIRCUMFERENCE,
        'G': MeasurementTypes.CIRCUMFERENCE,
        'H': MeasurementTypes.CIRCUMFERENCE,
        'I': MeasurementTypes.CIRCUMFERENCE,
        'J': MeasurementTypes.DISTANCE,
        'K': MeasurementTypes.HEIGHT,
        'L': MeasurementTypes.CIRCUMFERENCE,
        'M': MeasurementTypes.CIRCUMFERENCE,
        'N': MeasurementTypes.CIRCUMFERENCE,
        'O': MeasurementTypes.DISTANCE,
        'N': MeasurementTypes.HEIGHT
    }
    
    # Measurement type mapping used to call distance calculation function
    # for breadths, depths, and lengths.
    _MEASUREMENT_TYPE_MAPPING = {
        MeasurementTypes.BREADTH: MeasurementTypes.DISTANCE,
        MeasurementTypes.DEPTH: MeasurementTypes.DISTANCE,
        MeasurementTypes.LENGTH: MeasurementTypes.DISTANCE,
        MeasurementTypes.DISTANCE: MeasurementTypes.DISTANCE,
        MeasurementTypes.HEIGHT: MeasurementTypes.HEIGHT,
        MeasurementTypes.CIRCUMFERENCE: MeasurementTypes.CIRCUMFERENCE
    }
    
    def __init__(self, shape: np.ndarray, gender: str = 'neutral') -> None:
        self.gender = gender
        
        model = create_model(gender)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.faces = model.faces.squeeze()
        
        self.height = self._get_1d_measure('overall', 'height')
        self.verts = self.verts / self.height * self.AVG_HEIGHT[gender]

        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)
        self.volume = self.mesh.volume
        
    def get_body_measure(
            self, 
            name_or_label: str, 
            type: str = None
        ) -> float:
        ''' Measure body by name/label and measurement type.
        
            Parameters
            ----------
            name_or_label: str, mandatory
                The name (waist, hip, arm, ...) or label (A, B, C, ...)
                of the desired body measurement.
            type: str, optional
                The type of the body measurement (length, breadth, depth,
                or circumference). By default, it returns the most intuitive
                one (see `self._DEFAULT_TYPES`).
                
            Returns
            -------
            body_measurement: float
                Normalized measurement in cm (wrt to average height for 
                particular gender).
        '''
        type = self.DEFAULT_TYPES[type] if type is not None else type
        assert(type in MeasurementTypes.__dict__().values())
        
        if len(name_or_label) > 1:
            label = self.NAMES_TO_LABELS[name_or_label]
        elif len(name_or_label) == 1:
            label = name_or_label.upper()
        else:
            raise ValueError('Invalid name/label (empty string?)')
        assert(chr(label) >= chr('A') and chr(label) <= chr('O'))
        name = self.LABELS_TO_NAMES[label]
        
        return self._total_measure(name, type)
        
    @classmethod
    def _total_measure(
            self, 
            measure_name: str, 
            type: str
        ) -> float:
        ''' Measure given body measurement, wrt its type.
        
            Parameters
            ----------
            measure_name: str, mandatory
                The name (waist, hip, arm, ...) of the desired body measurement.
            type: str, optional
                The type of the body measurement (length, breadth, depth,
                or circumference). By default, it returns the most intuitive
                one (see `self._DEFAULT_TYPES`).
                
            Returns
            -------
            body_measurement: float
                Body measure in cm.
        '''
        idx_attr_name = f'{measure_name.upper()}_INDICES'
        _indexes = getattr(MeshIndices, idx_attr_name)
        
        dist_fun_type = self._MEASUREMENT_TYPE_MAPPING[type]
        _dist_fun = getattr(self, f'_get_{dist_fun_type}')
        
        line_segments = np.array([
            (self.verts[_indexes[idx]], self.verts[_indexes[idx+1]]) \
                for idx in range(len(_indexes) - 1)])
        return sum([_dist_fun([x[0], x[1]]) for x in line_segments])

    def all_measures(self, type: str = None) -> List[float]:
        ''' Get all body measurements (their common type is optional).
        
            Parameters
            ----------
            type: str, optional
                The type of the body measurement (length, breadth, depth,
                or circumference). By default, it returns the most intuitive
                one (see `self._DEFAULT_TYPES`).
                
            Returns
            -------
            body_measurement: List[float]
                Normalized body measurements in cm (wrt to the average 
                height for particular gender).
        '''
        return [self.measure(x, type) for x in self.all_names()]

    @staticmethod
    def _get_height(v1: List[float], v2: List[float]) -> float:
        return np.abs((v1 - v2))[1]
    
    @staticmethod
    def _get_dist(vs: List[float]) -> float:
        return np.linalg.norm(vs[0] - vs[1])

    @staticmethod
    def all_names() -> List[str]:
        return MeshMeasurements.LABELS_TO_NAMES.values()

    @staticmethod
    def all_labels() -> List[str]:
        return MeshMeasurements.NAMES_TO_LABELS.values()
    
    @staticmethod
    def all_full_names() -> List[str]:
        _all_names = MeshMeasurements.LABELS_TO_NAMES.values()
        corresponding_types = MeshMeasurements.DEFAULT_TYPES.values()
        return [f'{x}_{y}' for (x, y) in zip(_all_names, corresponding_types)]
