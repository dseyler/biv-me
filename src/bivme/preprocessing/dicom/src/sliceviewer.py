import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

import bivme.preprocessing.dicom.src.contouring as contouring
import bivme.preprocessing.dicom.src.guidepointprocessing as guidepointprocessing
import bivme.preprocessing.dicom.src.utils as utils

class SliceViewer:
    def __init__(self, processed_folder, slice_info_df, view, sliceID, es_phase, num_phases, full_cycle=True, my_logger=None):
        
        self.slice_info_df = slice_info_df
        self.view = view
        self.sliceID = sliceID
        self.es_phase = int(es_phase)
        self.num_phases = num_phases
        if full_cycle:
            self.phases = np.arange(0,self.num_phases, dtype=int)
        else:
            self.phases = [0, self.es_phase]
        self.slice = self.slice_info_df[(self.slice_info_df['View'] == self.view) & (self.slice_info_df['Slice ID'] == self.sliceID)]
        self.segmentation_folder = os.path.join(processed_folder, 'segmentations')
        self.segmentations = self.get_segmentations()
        self.get_contours()
        self.landmarks = None
        self.my_logger = my_logger
        
    def get_segmentations(self):
        segmentations = []

        segmentation = os.path.join(self.segmentation_folder, self.view, f"{self.view}_3d_{self.sliceID}.nii.gz")
        segmentation = nib.load(segmentation).get_fdata()
        segmentation = np.transpose(segmentation, (1, 0, 2))
        for i in range(0,segmentation.shape[2]):
            segmentations.append(segmentation[:,:,i])

        self.size = segmentations[0].shape
        return segmentations
    
    def get_initial_landmarks(self):
        self.get_landmarks_from_intersections()

        self.landmarks = {
            'SAX': {'RVI': {}},
            '2ch': {'MV': {}, 'LVA': {}}, 
            '3ch': {'MV': {}, 'AV': {}, 'LVA': {}},
            '4ch': {'MV': {}, 'TV': {}, 'LVA': {}},
            'RVOT': {'PV': {}}
        }

        if self.view == 'SAX':
            for phase in self.phases:
                self.landmarks['SAX']['RVI'][f'{phase}'] = self.rvi[self.view][f'{phase}']

        elif self.view == '2ch':
            for phase in self.phases:
                self.landmarks['2ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['2ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']
        
        elif self.view == '3ch':
            for phase in self.phases:
                self.landmarks['3ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['3ch']['AV'][f'{phase}'] = self.av[self.view][f'{phase}']
                self.landmarks['3ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']

        elif self.view == '4ch':
            for phase in self.phases:
                self.landmarks['4ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['4ch']['TV'][f'{phase}'] = self.tv[self.view][f'{phase}']
                self.landmarks['4ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']

        elif self.view == 'RVOT':
            for phase in self.phases:
                self.landmarks['RVOT']['PV'][f'{phase}'] = self.pv[self.view][f'{phase}']

        return self.landmarks

    def get_landmarks_from_intersections(self):

        self.rvi = {'SAX': {}}
        self.mv = {'2ch': {}, '3ch': {}, '4ch': {}}
        self.av = {'3ch': {}}
        self.tv = {'4ch': {}}
        self.lva = {'2ch': {}, '3ch': {}, '4ch': {}}
        self.pv = {'RVOT': {}}

        if self.view == 'SAX':
            for i, phase in enumerate(self.phases):
                try:
                    self.rvi['SAX'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 2, 3, distance_cutoff=2.5)
                except:
                    self.rvi['SAX'][f'{phase}'] = None

        elif self.view == '2ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['2ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 3, distance_cutoff=3)
                except:
                    self.mv['2ch'][f'{phase}'] = None

                try:
                    self.lva['2ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['2ch'][f'{phase}'][0],  self.mv['2ch'][f'{phase}'][1])
                except:
                    self.lva['2ch'][f'{phase}'] = None
        
        elif self.view == '3ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['3ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 4, distance_cutoff=3)
                except:
                    self.mv['3ch'][f'{phase}'] = None
                
                try:
                    self.av['3ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 5, distance_cutoff=3)
                except:
                    self.av['3ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Aortic valve not found on 3ch slice {self.sliceID} phase {phase}')

                try:
                    self.lva['3ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['3ch'][f'{phase}'][0],  self.mv['3ch'][f'{phase}'][1])
                except:
                    self.lva['3ch'][f'{phase}'] = None

        elif self.view == '4ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['4ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 4, distance_cutoff=3)
                except:
                    self.mv['4ch'][f'{phase}'] = None
                
                try:
                    self.tv['4ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 3, 5, distance_cutoff=3)
                except:
                    self.tv['4ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Tricuspid valve not found on 4ch slice {self.sliceID} phase {phase}')

                try:
                    self.lva['4ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['4ch'][f'{phase}'][0],  self.mv['4ch'][f'{phase}'][1])
                except:
                    self.lva['4ch'][f'{phase}'] = None

        elif self.view == 'RVOT':
            for i, phase in enumerate(self.phases):
                try:
                    self.pv['RVOT'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 3, distance_cutoff=3)
                except:
                    self.pv['RVOT'][f'{phase}'] = None
    
        else:
            self.my_logger.warning(f'Invalid view: {self.view}. Must be one of SAX, 2ch, 3ch, 4ch, or RVOT.')
        
    def get_contours(self):
        ## TODO: QC 
        contours = {}

        if self.view == 'SAX':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_SAX(self.segmentations[i])
        elif self.view == 'RVOT':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_RVOT(self.segmentations[i])
        elif self.view == '2ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_2ch(self.segmentations[i])
        elif self.view == '3ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_3ch(self.segmentations[i])
        elif self.view == '4ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_4ch(self.segmentations[i])
        
        self.contours = contours

    # TODO: Remove this - redundant
    def clean_lax(self):
        if self.view == '4ch':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                LV_endo = self.contours[str(phase)][0]
                LV_epi = self.contours[str(phase)][1]
                RV_endo = self.contours[str(phase)][0]
                RV_septal = self.contours[str(phase)][2]
                RV_epi = self.contours[str(phase)][1]


                MV_pts = self.mv[self.view][str(phase)]
                TV_pts = self.tv[self.view][str(phase)]

                # Clean between mitral valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][4] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][3] = LV_epi

                # Clean between tricuspid valve points
                del_idx = []
                for i in range(len(RV_endo)):
                    point = RV_endo[i]
                    valve_dist = np.linalg.norm(TV_pts[1] - TV_pts[0])
                    distance1 = np.linalg.norm(TV_pts[1] - point)
                    distance2 = np.linalg.norm(TV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_endo = np.delete(RV_endo, del_idx, axis=0)
                self.contours[str(phase)][0] = RV_endo

                del_idx = []
                for i in range(len(RV_septal)):
                    point = RV_septal[i]
                    valve_dist = np.linalg.norm(TV_pts[1] - TV_pts[0])
                    distance1 = np.linalg.norm(TV_pts[1] - point)
                    distance2 = np.linalg.norm(TV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_septal = np.delete(RV_septal, del_idx, axis=0)
                self.contours[str(phase)][2] = RV_septal

        elif self.view == '3ch':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                RV_endo = self.contours[str(phase)][0]
                RV_epi = self.contours[str(phase)][1]
                RV_septal = self.contours[str(phase)][2]
                LV_epi = self.contours[str(phase)][3]
                LV_endo = self.contours[str(phase)][4]

                MV_pts = self.mv[self.view][str(phase)]
                AV_pts = self.av[self.view][str(phase)]

                # Clean between mitral valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][4] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][3] = LV_epi

                # Clean between aortic valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(AV_pts[1] - AV_pts[0])
                    distance1 = np.linalg.norm(AV_pts[1] - point)
                    distance2 = np.linalg.norm(AV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][4] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(AV_pts[1] - AV_pts[0])
                    distance1 = np.linalg.norm(AV_pts[1] - point)
                    distance2 = np.linalg.norm(AV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)

                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][3] = LV_epi

        elif self.view == '2ch':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                LV_endo = self.contours[str(phase)][0]
                LV_epi = self.contours[str(phase)][1]

                MV_pts = self.mv[self.view][str(phase)]

                # Clean between mitral valve points
                del_idx = []
                for i in range(len(LV_endo)):
                    point = LV_endo[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_endo = np.delete(LV_endo, del_idx, axis=0)
                self.contours[str(phase)][0] = LV_endo

                del_idx = []
                for i in range(len(LV_epi)):
                    point = LV_epi[i]
                    valve_dist = np.linalg.norm(MV_pts[1] - MV_pts[0])
                    distance1 = np.linalg.norm(MV_pts[1] - point)
                    distance2 = np.linalg.norm(MV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                LV_epi = np.delete(LV_epi, del_idx, axis=0)
                self.contours[str(phase)][1] = LV_epi
        
        elif self.view == 'RVOT':
            # remove points that are not on the endocardial contours
            for phase in self.phases:
                RV_endo = self.contours[str(phase)][0]
                RV_septal = self.contours[str(phase)][1]

                PV_pts = self.pv[self.view][str(phase)]

                # Clean between pulmonary valve points
                del_idx = []
                for i in range(len(RV_endo)):
                    point = RV_endo[i]
                    valve_dist = np.linalg.norm(PV_pts[1] - PV_pts[0])
                    distance1 = np.linalg.norm(PV_pts[1] - point)
                    distance2 = np.linalg.norm(PV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_endo = np.delete(RV_endo, del_idx, axis=0)
                self.contours[str(phase)][0] = RV_endo

                del_idx = []
                for i in range(len(RV_septal)):
                    point = RV_septal[i]
                    valve_dist = np.linalg.norm(PV_pts[1] - PV_pts[0])
                    distance1 = np.linalg.norm(PV_pts[1] - point)
                    distance2 = np.linalg.norm(PV_pts[0] - point)
                    point_dist = distance1 + distance2

                    if abs(point_dist-valve_dist) < 3:
                        del_idx.append(i)
                
                RV_septal = np.delete(RV_septal, del_idx, axis=0)
                self.contours[str(phase)][1] = RV_septal

    def get_slice_info(self):
        self.imgPos = self.slice['ImagePositionPatient'].values[0]
        self.imgOrient = self.slice['ImageOrientationPatient'].values[0]
        self.ps = self.slice['Pixel Spacing'].values[0]


    def smooth_landmarks(self, landmarks):
        # Smooth landmarks by applying an low pass filter
        # Firstly, check if any of the landmarks are flipped across the cycle
        if np.any([x is None for x in landmarks]):
            return landmarks # No smoothing if there are missing landmarks
        
        l1 = np.array([x[0] for x in landmarks])
        l2 = np.array([x[1] for x in landmarks])

        true_l1 = []
        true_l2 = []

        # # Take the ED orientation as the true orientation
        l1_ed = l1[0]
        l2_ed = l2[0]

        for m, n in zip(l1, l2):
            if np.all(m == l1_ed) and np.all(n == l2_ed):
                # If the landmark is the same as the ED landmark, keep it
                true_l1.append(m)
                true_l2.append(n)

            else:
                # Check if the landmark is flipped
                if np.linalg.norm(m - previous_l1) > np.linalg.norm(n - previous_l1) or np.linalg.norm(n - previous_l2) > np.linalg.norm(m - previous_l2):
                    # If it is flipped, correct the orientation
                    true_l1.append(n)
                    true_l2.append(m)
                else:
                    true_l1.append(m)
                    true_l2.append(n)

            previous_l1 = true_l1[-1]
            previous_l2 = true_l2[-1]
        
        true_l1 = np.array(true_l1)
        true_l2 = np.array(true_l2)

        # Smooth the landmarks using a low pass filter
        harmonic_divisor = 4 # Low pass filter will keep num_phases/harmonic_divisor harmonics. E.g. if you have 30 frames and harmonic_divisor=4, it will keep the first 7 harmonics.

        l1_x = np.array([x[0] for x in true_l1])
        l1_y = np.array([x[1] for x in true_l1])
        l2_x = np.array([x[0] for x in true_l2])
        l2_y = np.array([x[1] for x in true_l2])

        l1_x_smooth = utils.apply_fft(l1_x, harmonic_divisor=harmonic_divisor)
        l1_y_smooth = utils.apply_fft(l1_y, harmonic_divisor=harmonic_divisor)
        l2_x_smooth = utils.apply_fft(l2_x, harmonic_divisor=harmonic_divisor)
        l2_y_smooth = utils.apply_fft(l2_y, harmonic_divisor=harmonic_divisor)

        l1_smooth = np.array([l1_x_smooth, l1_y_smooth]).T
        l2_smooth = np.array([l2_x_smooth, l2_y_smooth]).T

        # Reconstruct the landmarks
        smoothed_landmarks = []
        for i in range(len(l1_smooth)):
            if np.isnan(l1_smooth[i]).any() or np.isnan(l2_smooth[i]).any():
                smoothed_landmarks.append(None)
            else:
                smoothed_landmarks.append(np.array([l1_smooth[i], l2_smooth[i]]))

        return smoothed_landmarks
    
    def smooth_LVA(self, landmarks):
        # No need to correct for orientation, so just smooth
        harmonic_divisor = 4 # Low pass filter will keep num_phases/harmonic_divisor harmonics. E.g. if you have 30 frames and harmonic_divisor=4, it will keep the first 7 harmonics.

        l_x = np.array([x[0] for x in landmarks])
        l_y = np.array([x[1] for x in landmarks])

        l_x_smooth = utils.apply_fft(l_x, harmonic_divisor=harmonic_divisor)
        l_y_smooth = utils.apply_fft(l_y, harmonic_divisor=harmonic_divisor)

        l_smooth = np.array([l_x_smooth, l_y_smooth]).T

        # Reconstruct the landmarks
        smoothed_landmarks = [l_smooth[i] if not np.isnan(l_smooth[i]).any() else None for i in range(len(l_smooth))]

        return smoothed_landmarks

    def export_slice(self, output_folder, slice_mapping, smooth_landmarks):
        self.get_slice_info()
        os.makedirs(output_folder, exist_ok=True)

        self.mapped_sliceID = slice_mapping[self.sliceID]
        
        if self.view == 'SAX':
            if smooth_landmarks:
                # Get all landmarks
                landmarks = [self.rvi[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the rvi dict
                self.rvi[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.rvi[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][4]

                RVI_pts = self.rvi[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                                RV_epi_pts,
                                RV_septal_pts,
                                LV_epi_pts,
                                LV_endo_pts,
                                RVI_pts]
                
                labels = ['SAX_RV_FREEWALL',
                            'SAX_RV_EPICARDIAL',
                            'SAX_RV_SEPTUM',
                            'SAX_LV_EPICARDIAL',
                            'SAX_LV_ENDOCARDIAL',
                            'RV_INSERT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue
                    
                    if len(points) == 0:
                        continue
                        
                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]

                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

        elif self.view == 'RVOT':
            if smooth_landmarks:
                # Get all landmarks
                landmarks = [self.pv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the pv dict
                self.pv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.pv[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                RV_s_pts = self.contours[phase][0]
                RV_fw_pts = self.contours[phase][1]
                RV_epi_pts = self.contours[phase][2]
                pa_pts = self.contours[phase][3]

                PV_pts = self.pv[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                                RV_s_pts,
                                RV_epi_pts,
                                PV_pts]
                
                # labels = ['SAX_RV_FREEWALL',
                #             'SAX_RV_SEPTUM',
                #             'SAX_RV_EPICARDIAL',
                #             'PULMONARY_VALVE']
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_SEPTUM',
                            'LAX_RV_EPICARDIAL',
                            'PULMONARY_VALVE']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]
                            
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

        elif self.view == '2ch':
            if smooth_landmarks:
                ## Get MV landmarks
                landmarks = [self.mv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the mv dict
                self.mv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.mv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get LVA landmarks
                landmarks = [self.lva[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_LVA(landmarks)

                # Write the smoothed landmarks to the lva dict
                self.lva[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.lva[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                la_pts = self.contours[phase][2]

                MV_pts = self.mv[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [LV_endo_pts, 
                                LV_epi_pts,
                                MV_pts,
                                LVA_pts,
                                la_pts]
                
                labels = ['LAX_LV_ENDOCARDIAL',
                            'LAX_LV_EPICARDIAL',
                            'MITRAL_VALVE',
                            'APEX_POINT',
                            'LAX_LA']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    elif points.size == 2:
                        if points.shape != (2,):
                            points = points[0]

                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]

                    else:
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]
                    
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

        elif self.view == '3ch':
            if smooth_landmarks:
                ## Get MV landmarks
                landmarks = [self.mv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the mv dict
                self.mv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.mv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get AV landmarks
                landmarks = [self.av[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the av dict
                self.av[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.av[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get LVA landmarks
                landmarks = [self.lva[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_LVA(landmarks)

                # Write the smoothed landmarks to the lva dict
                self.lva[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.lva[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][6]

                la_pts = self.contours[phase][4]

                MV_pts = self.mv[self.view][str(phase)]
                AV_pts = self.av[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                               RV_epi_pts,
                               RV_septal_pts,
                               LV_epi_pts,
                               LV_endo_pts,
                               RV_epi_pts,
                               la_pts,
                               MV_pts,
                               AV_pts,
                               LVA_pts]
                
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_RV_SEPTUM',
                            'LAX_LV_EPICARDIAL',
                            'LAX_LV_ENDOCARDIAL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_LA',
                            'MITRAL_VALVE',
                            'AORTA_VALVE',
                            'APEX_POINT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points)== 0:
                        continue

                    elif points.size == 2:
                        if points.shape != (2,):
                            points = points[0]

                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]


                    else:
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]

                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=1.0)

        elif self.view == '4ch':
            if smooth_landmarks:
                ## Get MV landmarks
                landmarks = [self.mv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the mv dict
                self.mv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.mv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get TV landmarks
                landmarks = [self.tv[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the tv dict
                self.tv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.tv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get LVA landmarks
                landmarks = [self.lva[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_LVA(landmarks)

                # Write the smoothed landmarks to the lva dict
                self.lva[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.lva[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][6]

                la_pts = self.contours[phase][4]
                ra_pts = self.contours[phase][5]

                MV_pts = self.mv[self.view][str(phase)]
                TV_pts = self.tv[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                               RV_epi_pts,
                               RV_septal_pts,
                               LV_epi_pts,
                               LV_endo_pts,
                               RV_epi_pts,
                                la_pts,
                                ra_pts,
                               MV_pts,
                               TV_pts,
                               LVA_pts]
                
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_RV_SEPTUM',
                            'LAX_LV_EPICARDIAL',
                            'LAX_LV_ENDOCARDIAL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_LA',
                            'LAX_RA',
                            'MITRAL_VALVE',
                            'TRICUSPID_VALVE',
                            'APEX_POINT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    elif points.size == 2:
                        if points.shape != (2,):
                            points = points[0]

                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]

                    else:
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]
                        
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.mapped_sliceID, weight=1.0, phase=int(phase))

    def plot_slice(self):
        import matplotlib
        matplotlib.use('TkAgg')
        plt.ion()

        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        # image and segmentation for each time frame
        # ax[0, 0].imshow(self.images[0], cmap='gray')
        ax[0, 0].imshow(self.segmentations[0], cmap='inferno', alpha=0.2)
        ax[0, 0].set_title(f'{self.view} Slice {self.sliceID} Time Frame 0')
        ax[0, 0].axis('off')
        # ax[0, 1].imshow(self.images[1], cmap='gray')
        ax[0, 1].imshow(self.segmentations[1], cmap='inferno', alpha=0.2)
        ax[0, 1].set_title(f'{self.view} Slice {self.sliceID} Time Frame {int(self.es_phase)}')
        ax[0, 1].axis('off')

        # landmarks and contours
        ax[1, 0].imshow(self.images[0], cmap='gray')
        ax[1, 1].imshow(self.images[1], cmap='gray')
        colours = ['r', 'g', 'b', 'y', 'm']

        for i, phase in enumerate(self.contours):
            for j, contype in enumerate(self.contours[phase]):
                try:
                    ax[1, i].scatter(contype[:,0], contype[:,1], c=colours[j], s=1)
                except:
                    pass
        
        try:
            for i, phase in enumerate([0, int(self.es_phase)]):
                for j, landmarktype in enumerate(self.landmarks[self.view]):
                    try:
                        ax[1, i].scatter(self.landmarks[self.view][landmarktype][str(phase)][0][0], self.landmarks[self.view][landmarktype][str(phase)][0][1], c='white', marker = '+', s=5)
                        ax[1, i].scatter(self.landmarks[self.view][landmarktype][str(phase)][1][0], self.landmarks[self.view][landmarktype][str(phase)][1][1], c='white', marker = '+', s=5)
                    except:
                        try:
                            ax[1, i].scatter(self.landmarks[self.view][landmarktype][str(phase)][0], self.landmarks[self.view][landmarktype][str(phase)][1], c='white', marker = '+', s=5)
                        except:
                            pass
                        pass
        except:
            pass
        plt.show(block=False)
        plt.pause(0.01)


        

        
    


    