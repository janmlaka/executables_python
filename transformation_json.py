import numpy as np
import json
import math
import pandas as pd
from kart2proj import kart2proj_fun
from proj2kart import  GK2FLh, proj2kart_fun

scaleW = 180/math.pi*3600
scaleM = 1e6

def transformation_json_D48_D96(json_file_D48, json_file_D96): 

    """
        This function does one specific thing:
        Transforms coordinates from coordinate system or projections D48/GK to D96/TM (only for this).
        In future I hope to expand this project to transform from between any two coordinate systems.
        
        So what this function does is in simple non-geodetic terms transform coordinates from D48 to D96 coordinate systems.
        D48 coordinate system is in Slovenia referred to as the "old coordinate system", while D96 as the "new coordinate system".
        
        If you are interested, I also implemented it through Least Squares methods (LSM). 
    """

    def read_store_data(data_json_D48, data_json_D96):
        
        """
        Description: Reads data from 2 files:
            - D48
            - D96

        And then processes said data into a JSON dictionary.

        INPUT:
        read_data_D48 = input file (txt or any csv) with coordinates in D48 c.s.
        read_data_D96 = input file (txt or any csv) with coordinates in D96 c.s.

        OUTPUT:
        JSON dictionary of all control points.
        """

        file48 = pd.read_csv(data_json_D48, sep='\s+', header=None)
        df_D48 = pd.DataFrame(file48)
        df_D48.columns = ["point_id", "y", "x", "h48", "st_y", "st_x", "st_h48"]

        file96 = pd.read_csv(data_json_D96, sep='\s+', header=None)
        df_D96 = pd.DataFrame(file96)
        df_D96.columns = ["point_id", "e", "n", "h96", "st_e", "st_n", "st_h96"]

        json_dict = {"points": [
        ]}

        for row48, row96 in zip(df_D48.to_dict(orient="records"), df_D96.to_dict(orient="records")):
            json_dict["points"].append({"D48":row48, "D96":row96})

        return json_dict
    
    json_dict = read_store_data(json_file_D48, json_file_D96)

    def read_convert_data(json_dict):
        """
        Description: Converts projection coordinates into cartesian.

        INPUT:
            - json_dict ==> JSON dictionary contains all the coordinates from control points (and all their standart deviations).

        OUTPUT:
        Result of this function:

            - cart_48 ==> cartesian coordinates from D48 c.s.
            - cart_96 ==> cartesian coordinates from D96 c.s.

            - fi_la_48 ==> geodetic coordinates from D48 c.s.
            - fi_la_96 ==> geodetic coordinates from D96 c.s.

            - k ==> number of control points
            - json_dict ==> JSON dictionary with all the control points stored
        """
        k = len(json_dict["points"])
        cart_48 = np.zeros((k,3))

        fi_la_48 = np.zeros((k,2))
        #D48 cartesian data
        for i in range(k):
            x48 = json_dict["points"][i]["D48"]["x"]
            y48 = json_dict["points"][i]["D48"]["y"]
            h48 = json_dict["points"][i]["D48"]["h48"]

            [x_cart_48, y_cart_48, z_cart_48] = proj2kart_fun(y48, x48, h48, np.radians(15), 0.9999, 6377397.155, 0.08169683087477896,5000000, 500000)
            cart_48[i,:] = np.hstack([x_cart_48, y_cart_48, z_cart_48])

            fi_la_D48 = GK2FLh(y48,x48,np.radians(15), 0.9999, 6377397.155, 0.08169683087477896, 5000000, 500000)
            fi_la_48[i,:] = np.hstack(fi_la_D48)

        # lambda == 0, fi == 1 

        # D96 cartesian data
        cart_96 = np.zeros((k,3))
        fi_la_96 = np.zeros((k,2))
        for i in range(k):
            n96 = json_dict["points"][i]["D96"]["n"]
            e96 = json_dict["points"][i]["D96"]["e"]
            h96 = json_dict["points"][i]["D96"]["h96"]

            [x_cart_96, y_cart_96, z_cart_96] = proj2kart_fun(e96, n96, h96, np.radians(15), 0.9999, 6378137, 0.08181919104281514, 5000000, 500000)
            cart_96[i,:] = np.hstack([x_cart_96, y_cart_96, z_cart_96]) 

            fi_la_D96 = GK2FLh(e96,n96,np.radians(15), 0.9999, 6378137, 0.08181919104281514, 5000000, 500000)
            fi_la_96[i,:] = np.hstack(fi_la_D96)

        return cart_48, cart_96, fi_la_48, fi_la_96, json_dict, k
    
    cart_48, cart_96, fi_la_48, fi_la_96, json_dict, k = read_convert_data(json_dict)

    def adjustment(cart_48, cart_96, fi_la_48, fi_la_96, k, json_dict):

        """
        Description: 
        The adjustment is basically the most important step (and the longest). Here is where all the calculations are made. They are made in a iterative process.

        INPUT:

            - cart_48 ==> cartesian coordinates from D48 c.s.
            - cart_96 ==> cartesian coordinates from D96 c.s.

            - fi_la_48 ==> geodetic coordinates from D48 c.s.
            - fi_la_96 ==> geodetic coordinates from D96 c.s.

            - k ==> number of control points
            - json_dict ==> JSON dictionary with all the control points stored 

        ADDITIONAL DESCRIPTION:

        It takes all the previous outputs and uses them to form the following:

            - vector f (basically contains the difference between D48 coordinates and D96)
            - Matrix A (derivatives along the measurements, which are coordinates in D48 and D96 c.s.)
            - Matrix B (this matrix contrains all the derivatives from all the unknowns (transformation parameters - there is 7 of them))
            - All the coordinates are cartesian in this step (except for standart deviations / variances):
                - That is why we convert the standart deviations (s.d.) from projected s.d. into cartesian s.d.
                - Basically the rule here is everything must be cartesian, in order to not mix up projected and cartesian data.

        OUTPUT:

            Final step:
            Calculate the following:
                - Functional model (which we already did :) ==> matrix A and B, vector f),
                - Stohastic model (standart deviations into weights),
                - Standart deviations for (transformation parameters, all the input data (D48 and D96 files), corrections),
                - Reference variance a-posteriori.
        """

        tx = 0
        ty = 0
        tz = 0
        wx = 0
        wy = 0
        wz = 0
        m = 1

        X0 = np.array((tx, ty, tz, wx, wy, wz, m)).T

        residual = 1

        n = 6*k
        u = len(X0) # 7
        n0 = 3*k + u
        r = n - n0
        c = r + u

        ni = 1

        while residual > 1e-6:
            Rx = np.array(([1.0, 0.0, 0.0],
                    [0.0, math.cos(X0[3]), -math.sin(X0[3])],
                    [0.0, math.sin(X0[3]), math.cos(X0[3])]), dtype=float)
        
            Ry = np.array(([math.cos(X0[4]), 0.0, math.sin(X0[4])],
                        [0.0, 1.0, 0.0],
                        [-math.sin(X0[4]), 0.0, math.cos(X0[4])]), dtype=float)
            
            Rz = np.array(([math.cos(X0[5]), -math.sin(X0[5]), 0.0],
                        [math.sin(X0[5]), math.cos(X0[5]), 0.0],
                        [0.0, 0.0, 1.0]), dtype=float)
            
            R = Rx @ Ry @ Rz
            
            T = np.array((tx, ty, tz), dtype= float).T
            T = T.reshape(3,1)
            
            f = np.zeros((c,1))
            for i in range(k):
                cart_48_i = np.array(cart_48[i]).reshape(3,1)
                cart_96_i = np.array(cart_96[i]).reshape(3,1)
                Fi = T + m * (R @ (cart_48_i)) - (cart_96_i)
                Fi = np.array(Fi).reshape(3,1)
                f[i*3:i*3+3, :] = Fi
            f = -f

            A = np.zeros((c,n))
            for i in range(k):
                Fi_D48 = m*R
                Fi_D96 = -np.eye(3)
                Fi_element = np.hstack((Fi_D48, Fi_D96))
                A[i*3:i*3+3, i*6:i*6+6] = Fi_element

            Rx_wx = np.array([[0.0, 0.0, 0.0], 
                            [0.0, -np.sin(X0[3]), -np.cos(X0[3])], 
                            [0.0, np.cos(X0[3]), -np.sin(X0[3])]], dtype='float64')
            
            Ry_wy = np.array([[-np.sin(X0[4]), 0.0, np.cos(X0[4])], 
                                            [0.0, 0.0, 0.0], 
                            [-np.cos(X0[4]), 0.0, -np.sin(X0[4])]], dtype='float64')
            
            Rz_wz = np.array([[-np.sin(X0[5]), -np.cos(X0[5]), 0.0], 
                            [np.cos(X0[5]), -np.sin(X0[5]), 0.0], 
                            [0.0, 0.0, 0.0]], dtype='float64')

            B = np.zeros((c,u))
            for i in range(k):
                F_T = np.eye(3)
                F_wx = np.array((m * (Rz @ Ry @ Rx_wx @ cart_48[i])) / scaleW)
                F_wx = F_wx.reshape(3,1)
                F_wy = np.array((m * (Rz @ Ry_wy @ Rx @ cart_48[i])) / scaleW)
                F_wy = F_wy.reshape(3,1)
                F_wz = np.array((m * (Rz_wz @ Ry @ Rx @ cart_48[i])) / scaleW)
                F_wz = F_wz.reshape(3,1)
                F_m = np.array((R @ cart_48[i]) / scaleM)
                F_m = F_m.reshape(3,1)
                F_unknowns = np.hstack((F_T, F_wx, F_wy, F_wz, F_m))
                B[i*3:i*3+3, 0:u] = F_unknowns

            # WEIGHTS LOGIC
            SG = np.zeros((6,6))
            S = np.zeros((n,n))
            
            for i in range(k):
                var_x48 = float((json_dict["points"][i]["D48"]["st_x"]))**2
                var_y48 = float((json_dict["points"][i]["D48"]["st_y"]))**2
                var_h48 = float((json_dict["points"][i]["D48"]["st_h48"]))**2

                var_n96 = float((json_dict["points"][i]["D96"]["st_n"]))**2
                var_e96 = float((json_dict["points"][i]["D96"]["st_e"]))**2
                var_h96 = float((json_dict["points"][i]["D96"]["st_h96"]))**2
                
                # lambda == 0, fi == 1 

                la_d48_i = fi_la_48[i,0] 
                fi_d48_i = fi_la_48[i,1]

                la_d96_i = fi_la_96[i,0]
                fi_d96_i = fi_la_96[i,1]

                var_ele_D48 = np.array((var_x48, var_y48, var_h48))
                var_ele_D96 = np.array((var_n96, var_e96, var_h96))

                Sr_D48 = np.zeros((3,3))
                Sr_D96 = np.zeros((3,3))

                np.fill_diagonal(Sr_D48, var_ele_D48)
                np.fill_diagonal(Sr_D96, var_ele_D96)
                
                J_D48 = np.array([[-math.sin(fi_d48_i)*math.cos(la_d48_i), -math.sin(la_d48_i), math.cos(fi_d48_i)*math.cos(la_d48_i)],
                [-math.sin(fi_d48_i)*math.sin(la_d48_i), math.cos(la_d48_i), math.cos(fi_d48_i)*math.sin(la_d48_i)],
                            [math.cos(fi_d48_i),        0,                math.sin(fi_d48_i)]])

                J_D96 = np.array([[-math.sin(fi_d96_i)*math.cos(la_d96_i), -math.sin(la_d96_i), math.cos(fi_d96_i)*math.cos(la_d96_i)],
                [-math.sin(fi_d96_i)*math.sin(la_d96_i), math.cos(la_d96_i), math.cos(fi_d96_i)*math.sin(la_d96_i)],
                            [math.cos(fi_d96_i),        0,                math.sin(fi_d96_i)]])

                SG_D48 = J_D48 @ Sr_D48 @ J_D48.T
                SG_D96 = J_D96 @ Sr_D96 @ J_D96.T

                SG[0:3, 0:3] = SG_D48
                SG[3:6, 3:6] = SG_D96

                S[i*6:i*6+6, i*6:i*6+6] = SG

            var_apriori = np.mean(np.mean(S))
            Q = (1/var_apriori) * S
            # Q = np.eye(n)
            P = np.linalg.inv(Q)

            Qe = A @ Q @ A.T
            Pe = np.linalg.inv(Qe)

            N = B.T @ Pe @ B
            t = B.T @ Pe @ f

            delta = np.linalg.inv(N) @ t 

            tx = X0[0] + delta[0] 
            tx = tx[0]
            ty = X0[1] + delta[1]
            ty = ty[0]
            tz = X0[2] + delta[2]
            tz = tz[0]
            
            wx = X0[3] + (delta[3] / scaleW)
            wx = wx[0]
            wy = X0[4] + (delta[4] / scaleW)
            wy = wy[0]
            wz = X0[5] + (delta[5] / scaleW)
            wz = wz[0]
            
            m = X0[6] + (delta[6] / scaleM)
            m = m[0]


            X0 = np.array([tx, ty, tz, wx, wy, wz, m]).T
            X = np.array([tx, ty, tz, wx, wy, wz, m]).T
            

            v = Q @ A.T @ Pe @ (f-B @ delta)

            residual = np.linalg.norm(delta)

            ni += 1
            if ni >= 15:
                break

        X = np.array([tx, ty, tz, wx, wy, wz, m]).T

        tx = X[0]
        ty = X[1]
        tz = X[2]

        wx = X[3]
        wy = X[4]
        wz = X[5]

        m = X[6]

        Rx = np.array(([1.0, 0.0, 0.0],
                        [0.0, math.cos(wx), -math.sin(wx)],
                        [0.0, math.sin(wx), math.cos(wx)]), dtype=float)
            
        Ry = np.array(([math.cos(wy), 0.0, math.sin(wy)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(wy), 0.0, math.cos(wy)]), dtype=float)

        Rz = np.array(([math.cos(wz), -math.sin(wz), 0.0],
                        [math.sin(wz), math.cos(wz), 0.0],
                        [0.0, 0.0, 1.0]), dtype=float)

        R = Rx @ Ry @ Rz

        T = np.array((tx, ty, tz), dtype=float).T

        ref_var_a_post = (v.T @ P @ v)/r
        ref_std = math.sqrt(ref_var_a_post)

        #natan?nost neznank
        Qdd = np.linalg.inv(N)
        #natan?nost popravkov
        Qvv = Q @ A.transpose() @ Pe @(np.eye(c) - B @ Qdd @ B.transpose() @ Pe)@ A @ Q
        #natan?nost izravnanih koli?in
        QLi = Q - Qvv

        #natancnost neznank - transformacijski parametri
        Sdd = ref_var_a_post * Qdd

        Std_tx = (math.sqrt(Sdd[0,0]))
        Std_ty = (math.sqrt(Sdd[1,1]))
        Std_tz = (math.sqrt(Sdd[2,2]))
        Std_wx = (math.sqrt(Sdd[3,3]))
        Std_wy = (math.sqrt(Sdd[4,4])) 
        Std_wz = (math.sqrt(Sdd[5,5]))
        Std_m = (math.sqrt(Sdd[6,6]))

        Svv = ref_var_a_post * Qvv
        SLi = ref_var_a_post * QLi

        adjustment_dict = {
            "T": T,
            "R": R,
            "m": m,
            "X": X,
            "Sdd": Sdd
        }

        standart_deviations = [Std_tx, Std_ty, Std_tz, Std_wx, Std_wy, Std_wz, Std_m]

        return adjustment_dict, cart_48, X, standart_deviations
    
    adjustment_dict, cart_48, X, standart_deviations = adjustment(cart_48, cart_96, fi_la_48, fi_la_96, k, json_dict)

    def transformation(k, adjustment_dict, cart_48):

        """
        This function as you might imagine transforms from D48 to D96.

        INPUT:

            - k ==> number of control points (integer)
            - adjustement_dict ==> contains all the transformation parameters and their standart deviations (matrix Sdd).
            - cart_48 ==> contains all the data from the c.s. D48.

        OUTPUT:

            - trans_coo_proj_array ==> transformed projection coordinates in c.s. D96.
            

        """

        T = adjustment_dict["T"]
        R = adjustment_dict["R"]
        m = adjustment_dict["m"]

        trans_coo_proj_array = np.zeros((k,3))
        for i in range(k):
            trans_cart_coo = T + (m * (R @ cart_48[i]))
            projekcijske_coo = np.array([kart2proj_fun(trans_cart_coo[0], trans_cart_coo[1], trans_cart_coo[2],
                                            np.radians(15), 0.9999, 6378137.0, 0.08181919104281514, 5000000, 500000)])
            projekcijske_coo = np.round(projekcijske_coo, 3)
            trans_coo_proj_array[i, :] = np.hstack((projekcijske_coo))

        return trans_coo_proj_array
    
    trans_coo_proj_array = transformation(k, adjustment_dict, cart_48)
    
    def residuals_statistics(k, json_dict, trans_coo_proj_array):
        residual_n_li = []
        residual_e_li = []
        residual_h_li = []
        for i in range(k):
            residual_n = json_dict["points"][i]["D96"]["n"] - trans_coo_proj_array[i, 0]
            residual_n_li.append(residual_n)

            residual_e = json_dict["points"][i]["D96"]["e"] - trans_coo_proj_array[i, 1]
            residual_e_li.append(residual_e)

            residual_h = json_dict["points"][i]["D96"]["h96"] - trans_coo_proj_array[i, 2]
            residual_h_li.append(residual_h)

        for i in range(k):
            mean_n = np.mean(residual_n_li)
            mean_e = np.mean(residual_e_li)
            mean_h = np.mean(residual_h_li)

            std_n = np.std(residual_n_li)
            std_e = np.std(residual_e_li)
            std_h = np.std(residual_h_li)

        res_n2_li = []
        res_e2_li = []
        res_h2_li = []

        for i in range(k):
            residual_n2 = residual_n_li[i]**2
            res_n2_li.append(residual_n2)

            residual_e2  = residual_e_li[i]**2
            res_e2_li.append(residual_e2)

            residual_h2 = residual_h_li[i]**2
            res_h2_li.append(residual_h2)

        mse_n = np.mean(res_n2_li)
        mse_e = np.mean(res_e2_li)
        mse_h = np.mean(res_h2_li)

        rmse_n = np.sqrt(mse_n)
        rmse_e = np.sqrt(mse_e)
        rmse_h = np.sqrt(mse_h)

        stats = [mean_n, mean_e, mean_h, std_n, std_e, std_h, rmse_n, rmse_e, rmse_h]

        return stats
            

    stats = residuals_statistics(k, json_dict, trans_coo_proj_array)

    def report(json_dict, k, trans_coo_proj_array, X, standart_deviations, stats):
        """
        Generates a report which includes:
            - Transformation parameters
            - Standart deviations for Transformation parameters
            - Transformed coordinates

        In future I will try to add some more parameters to this function.
        """
        with open("data.json", "w", encoding="utf-8") as file_json: # these 2 lines create a json file 
            json.dump(json_dict, file_json, indent= 2, ensure_ascii=False)


        with open("results_json.txt", "w") as results:
            results.write("Transformation parameters: \n") 

            results.write(f"Tx = {X[0]:.3f}" + " m" + "\n")
            results.write(f"Ty = {X[1]:.3f}" + " m" + "\n")
            results.write(f"Tz = {X[2]:.3f}" + " m" + "\n")

            results.write(f"Wx = {np.degrees(X[3])*3600:.6f}" + " sec" + "\n")
            results.write(f"Wy = {np.degrees(X[4])*3600:.6f}" + " sec" + "\n")
            results.write(f"Wz = {np.degrees(X[5])*3600:.6f}" + " sec" + "\n")

            results.write(f"m = {X[6]:.6f}" + "\n")
                
            results.write("\nStandart deviations for transformation parameters: \n")
            results.write(f"Std tx = {standart_deviations[0]:.6f}"  + "\n")
            results.write(f"Std ty = {standart_deviations[1]:.6f}"  + "\n")
            results.write(f"Std tz = {standart_deviations[2]:.6f}"  + "\n")

            results.write(f"Std wx = {standart_deviations[3]:.6f}"  + "\n")
            results.write(f"Std wy = {standart_deviations[4]:.6f}"  + "\n")
            results.write(f"Std wz = {standart_deviations[5]:.6f}"  + "\n")

            results.write(f"Std m = {standart_deviations[6]:.6f}"  + "\n")

            results.write("\nTransformed coordinates: \n")
            results.write("\tn\t\t \te\t\t h\n")
            for i in range(k):
                results.write(str(np.round(trans_coo_proj_array[i, 0], 3)) + "\t")
                results.write(str(np.round(trans_coo_proj_array[i, 1], 3)) + "\t")
                results.write(str(np.round(trans_coo_proj_array[i, 2], 3)) + "\n")

            results.write("\nStatistics: \n")
            results.write(f"Mean n = {stats[0]:.6f}" + "\n")
            results.write(f"Mean e = {stats[1]:.6f}" + "\n")
            results.write(f"Mean h = {stats[2]:.6f}" + "\n")

            results.write(f"Standard deviation n = {stats[3]:.6f}" + "\n")
            results.write(f"Standard deviation e = {stats[4]:.6f}" + "\n")
            results.write(f"Standard deviation h = {stats[5]:.6f}" + "\n")

            results.write(f"RMSE n = {stats[6]:.6f}" + "\n")
            results.write(f"RMSE e = {stats[7]:.6f}" + "\n")
            results.write(f"RMSE h = {stats[8]:.6f}" + "\n")

    report(json_dict, k, trans_coo_proj_array, X, standart_deviations, stats)

    

transformation_json_D48_D96("data_D48.txt", "data_D96.txt")





