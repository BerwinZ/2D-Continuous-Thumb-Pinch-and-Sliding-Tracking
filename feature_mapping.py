"""Feature mappings
"""
import math
from math import tan, sin, cos, pi
from math import degrees as degs
from math import radians as rads

import numpy as np
from scipy.optimize import fsolve

# ---------------------------------------------------------------
# Calibration parameters for imaging model
# ---------------------------------------------------------------

aph0 = rads(70)      # degree between camera x-axis and second knuckle 
theta4 = rads(120)   # degree of the fingertip of thumb

pt_a0 = [-28.17, 0]   # coord of point A0
d0 = 19.4       # length from point A0 to A1 (mm)
d1 = 25.0       # length of first knuckle of thumb (mm)
d2 = 35.0       # length of second knuckle of thumb (mm)
e  = 7.5        # half length of tangent plane (mm)

av = rads(39.2)   # vertical field of view
ah = rads(50.79)  # horizontal field of view
f  = 3.37                   # Focal length (mm)

resolution_x = 640          # horizontal pixels
resolution_y = 480          # vertical pixels
diag_len_sensor = 4         # diagonal length of sensor (mm)
kppi = 1 / (math.sqrt(resolution_x*resolution_x + resolution_y*resolution_y) / diag_len_sensor)            # from pixel coord to real len

class ImmMapping:
    def __init__(self):
        """Use imaging model to map features
        """
        self.aph1 = None
        self.aph2 = None
        self.beta = None
        
        self.pt_a1 = [pt_a0[0] + d0 * cos(aph0), pt_a0[1] + d0 * sin(aph0)]
        self.pt_a2 = [pt_a0[0] - (d2 - d0) * cos(aph0), 
                      pt_a0[1] - (d2 - d0) * sin(aph0)]
        
        self.c_state = 'origin'
        self.aph1_0 = 0.2109284390443787   # require calibration
        
        self.beta1 = 0
        self.beta2 = 11 * pi / 90
        self.g1 = 2.555
        self.g2 = 3.195

        # self.test1()
        # self.test2()

    def test1(self):
        for i in np.arange(0, 90, 0.5):
            rad_val = rads(i)
            b_val = self.__func_g(rad_val)
            print(i, round(degs(self.__func_g_inv(b_val)), 1), b_val)
    
    def test2(self):
        for i in np.arange(-22, 22, 0.5):
            rad_val = rads(i)
            g_val = self.__func_h(rad_val)
            print(i, round(degs(self.__func_h_inv(g_val)), 1), g_val)

    def calibrate(self, x_gim, y_bim):
        print('Current calibration state:', self.c_state)
        if self.c_state == 'origin':
            self.aph1_0 = self.__func_g_inv(kppi * y_bim)
            self.g1 = kppi * x_gim
            print(self.aph1_0, self.g1)
            self.c_state = 'left'
        elif self.c_state == 'left':
            self.g2 = kppi * x_gim
            print(self.g2)
            self.c_state = 'origin'

    def predict(self, x_gim, y_bim):
        if x_gim is None or y_bim is None:
            return None

        y = self.__pixel_to_coord_y(y_bim)
        x = self.__pixel_to_coord_x(x_gim)
        # print(x)
        # print(y)

        return x, y
    

    def __pixel_to_coord_x(self, xgim):
        self.beta = self.__func_h_inv(kppi * xgim) 
        print(kppi * xgim, degs(self.beta))
        return (d1 * sin(aph0 - self.aph1) + d2 * sin(aph0)) * self.beta

    def __pixel_to_coord_y(self, ybim):
        k = 4
        self.aph1 = self.__func_g_inv(k * kppi * ybim)
        # print('b=', k * kppi * ybim, 'deg=', degs(self.aph1))
        return d1 * (self.aph1 - self.aph1_0)

    def __func_g_inv(self, b):
        """aph1 = g^(-1) (b)

        Arguments:
            b {[type]} -- imaging length (2.5 - 7.2)

        Returns:
            aph1 -- rads of first knuckle (0 - pi/2)
        """
        equ = lambda x: self.__func_g(x) - b

        # There are 3 regions. x < 12, 12 <= x < 70, 70 <= x, so set different start state
        start_degs = [10, 40, 80, 85, 90]
        result_degs = []
        for start_deg in start_degs:
            result_degs.append(degs(fsolve(equ, rads(start_deg))[0]))

        result_deg = 74     # the range of max degs results

        # Select the result positive and is nearest to its round
        diff = float('inf')
        for val in result_degs:
            if val < 0:
                continue
            
            tmp_diff = abs(val - round(val, 3))
            if tmp_diff < diff:
                diff = tmp_diff
                result_deg = val

        # print(result_degs)
        return rads(result_deg)

    def __func_h_inv(self, g):
        """beta = h^(-1) (g)

        Arguments:
            g {[type]} -- imaging length (1.5 - 3)

        Returns:
            beta -- rads of second knuckle (-11pi/90 - 11pi/90)
        """
        equ = lambda x: self.__func_h(x) - g
        start_deg = 0
        return fsolve(equ, rads(start_deg))[0]

    def __func_g(self, aph1):
        """b = g(aph1)

        Arguments:
            aph1 {[type]} -- rads of first knuckle (0 - pi/2)

        Returns:
            b -- imaging length (2.5 - 7.2)
        """

        cot = lambda x: 1 / tan(x)
        pt_a3 = [0, 0]
        pt_a4 = [0, 0]
        pt_a5 = [0, 0]
        pt_c  = [0, 0]

        pt_a3[0] = self.pt_a1[0] + e * cos(aph1 - aph0)
        pt_a3[1] = self.pt_a1[1] + e * sin(aph1 - aph0)
        pt_a4[0] = self.pt_a1[0] + d1 * cos(aph0 - aph1)
        pt_a4[1] = self.pt_a1[1] + d1 * sin(aph0 - aph1)
        base_a5 = tan(aph0 - aph1 - theta4) - tan(aph0 - aph1)
        pt_a5[0] = (tan(aph0-aph1-theta4) * pt_a4[0] - tan(aph0-aph1) * pt_a3[0] - pt_a4[1] + pt_a3[1]) / base_a5
        pt_a5[1] = (tan(aph0-aph1-theta4) * tan(aph0-aph1) * (pt_a4[0]-pt_a3[0]) + tan(aph0-aph1-theta4) * pt_a3[1] - tan(aph0 - aph1) * pt_a4[1]) / base_a5

        pt_c[0] = (tan(aph0 - aph1) * pt_a3[0] - pt_a3[1]) / (tan(aph0 - aph1) + cot(av / 2))
        pt_c[1] = (tan(aph0 - aph1) * pt_a3[0] - pt_a3[1]) / (tan(aph0 - aph1) + cot(av / 2)) * cot(av / 2)

        # l1 = pt_c[0]  * pt_a5[1] - pt_a5[0] * pt_c[1]
        # l2 = pt_c[0]  * pt_a4[1] - pt_a4[1] * pt_c[1]
        # l3 = pt_a5[0] * pt_a4[1] - pt_a4[0] * pt_a5[1]

        # delta = l2 * l2 - 4 * l1 * l3
        # if delta >= 0:
        #     t1 = (2 * l1 - l2 + math.sqrt(delta)) / (2 * (l1 - l2 + l3))
        #     t2 = (2 * l1 - l2 - math.sqrt(delta)) / (2 * (l1 - l2 + l3))
        # else:
        #     t1 = 0
        #     t2 = 0

        func_L = lambda t: ((1-t) ** 2 * pt_c[1] + 2 * t * (1-t) * pt_a5[1] + t * t * pt_a4[1]) / ((1-t) ** 2 * pt_c[0] + 2 * t * (1-t) * pt_a5[0] + t * t * pt_a4[0])

        t_list = np.arange(0, 1, 0.1)
        L_t_list = func_L(t_list)
        if max(L_t_list > 0):
            L_t_list = L_t_list[L_t_list > 0]    
        L_t_star = min(L_t_list)

        # L_t_star = min(func_L(0), func_L(1))
        # if t1 > 0 and t1 < 1:
        #     L_t_star = min(L_t_star, func_L(t1))
        # if t2 > 0 and t2 < 1:
        #     L_t_star = min(L_t_star, func_L(t2))

        return f * (1 / L_t_star + math.tan(av/2))
    
    def __func_h(self, beta):
        """g = h(beta)

        Arguments:
            beta {[type]} -- rads of second knuckle (-11pi/90 - 11pi/90)

        Returns:
            g -- imaging length (1.5 - 3)
        """
        psi1 = f * tan(ah / 2) 
        psi2 = (d2 - d0) * cos(aph0)

        om = (f*f + psi1*psi1 + self.g1*self.g2 - self.g1*psi1 - self.g2*psi1) * sin(self.beta1 - self.beta2) + (f*self.g2 - f*self.g1) * cos(self.beta1 - self.beta2)

        xg = f*psi2 * ((self.g1-psi1)*cos(self.beta1)*sin(self.beta2) - (self.g2-psi1)*cos(self.beta2)*sin(self.beta1)) / om
        yg = -psi2*(psi1*psi1 + self.g1*self.g2  - self.g1*psi1 - self.g2*psi1) * sin(self.beta1 - self.beta2) / om + psi2 * 2 * f * (self.g1 - self.g2) * cos(self.beta1) * cos(self.beta2) / om

        return psi1 + f * (yg * tan(beta) - xg) / (xg * tan(beta) + yg + psi2)
       

class MknnMapping:
    pass

class MlrmMapping:
    def __init__(self, model_path_x, model_path_y):
        """Use random forest regression model to map features
        """
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        # Model
        print('-' * 60)
        print('Start Loading Model...')

        self.model_x = joblib.load(model_path_x)
        self.model_y = joblib.load(model_path_y)

        print(self.model_x.get_params())
        print(self.model_y.get_params())
        print("\nLoad model successfully!")

    def predict(self, features):
        if features is None:
            return None

        features = features.reshape(-1, 20)
        x = self.model_x.predict(features)[0]
        y = self.model_y.predict(features)[0]

        return (x, y)
        