import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import pandas as pd
import os


def generate_data(n, S):
        
    m = np.array([2, .4, 0, 0, .3, .1])
    
    df = pd.DataFrame(data = np.random.multivariate_normal(mean = m, cov = S, size = n),
                      columns = ["A1", "B1", "X1", "X2", "U2", "V2"])
    
    df["A2"] = df["A1"] + df["U2"]
    df["B2"] = df["B1"] + df["V2"]
    df["Y1"] = df["A1"] + df["B1"]*df["X1"]
    df["Y2"] = df["A2"] + df["B2"]*df["X2"]
     
    Sxx1 = np.linalg.inv(S[2:4, 2:4])
    Sab = S[0:2, 0:2]
    Sabx = S[0:2, 2:4]
    Sab_x = Sab - Sabx.dot(Sxx1).dot(Sabx.T) 
    truth = {"medEA": m[0], 
             "medEB": m[1], 
             "medVA": Sab_x[0,0], 
             "medVB": Sab_x[1,1], 
             "medCAB": Sab_x[0,1]}
        
    return df, truth



def get_ycmoms(df, g = 10):
    
    #n = df.shape[0] 
    K1 = pairwise_kernels(df[["X1", "X2"]], metric = "rbf", gamma = g, n_jobs = 1)
    K2 = pairwise_kernels(df[["X1", "X2"]], Y = df[["X1", "X1"]], metric = "rbf", gamma = 2*g, n_jobs = 1)
    
    fitter1 = lambda y: np.sum(K1*y.values.reshape(-1, 1), 0)/np.sum(K1, 0)
    fitter2 = lambda y: np.sum(K2*y.values.reshape(-1, 1), 0)/np.sum(K2, 0)
    
    ycmoms = {}
    ycmoms["EY1"] = fitter1(df["Y1"])
    ycmoms["EY2"] = fitter1(df["Y2"])
    ycmoms["VY1"] = fitter1((df["Y1"] - ycmoms["EY1"])**2)
    ycmoms["VY2"] = fitter1((df["Y2"] - ycmoms["EY2"])**2)
    ycmoms["C12"] = fitter1((df["Y1"] - ycmoms["EY1"])*(df["Y2"] - ycmoms["EY2"]))
    ycmoms["EDY"] = fitter2(df["Y2"] - df["Y1"])
    ycmoms["VDY"] = fitter2((df["Y2"] - df["Y1"] - ycmoms["EDY"])**2)
    
    return pd.DataFrame(ycmoms)



def get_abcmoms(df, ycmoms):
    
    g1 = lambda x: np.linalg.inv(np.array([[1, x[0]], 
                                           [1, x[1]]]))
    g2 = lambda x: np.linalg.inv(np.array([[1, x[0]**2,   2*x[0]], 
                                           [1, x[1]**2,   2*x[1]], 
                                           [1, x[0]*x[1], x[0]+x[1]]]))
    
    X = df[["X1", "X2"]].values
    
    EY = np.vstack([ycmoms["EY1"], 
                    ycmoms["EY2"] - ycmoms["EDY"]]).T
    
    CY = np.vstack([ycmoms["VY1"], 
                    ycmoms["VY2"] - ycmoms["VDY"],
                    ycmoms["C12"]]).T
    
    mab = []; sab = []
    for x,ey,cy in zip(X, EY, CY):
        mab += [ np.dot(g1(x), ey.reshape(-1, 1))]
        sab += [ np.dot(g2(x), cy.reshape(-1, 1))]

    abcmoms = pd.DataFrame(data = np.squeeze(np.hstack([mab, sab])),
                           columns = ["EAx", "EBx", "VAx", "VBx", "CABx"])
    return abcmoms



def get_medians(df, abcmoms, threshold):
    diff = np.abs(df["X2"] - df["X1"])
    discarding_output = abcmoms[diff > threshold].median()
    abcmoms[diff < threshold] = 0
    zeroing_output = abcmoms.median()
    return discarding_output, zeroing_output



if __name__ == "__main__":
    
    n = np.random.choice([500, 2000, 5000, 10000, 20000])
    g = np.random.choice([0.1, 1, 5, 10])
    scenario = np.random.randint(1, 7)
    
    if scenario == 1:
        S = np.array([[3, 1.5, 1, 1, 0, 0],
                       [1.5, 3, 1, 1, 0, 0],
                       [1, 1, 3, 1.5, 0, 0],
                       [1, 1, 1.5, 3, 0, 0],
                       [0, 0, 0, 0, .2, 0.1],
                       [0, 0, 0, 0, 0.1, .2]])
    elif scenario == 2:
        S = np.array([[6, 3, 2, 1, 0, 0],
                      [3, 6, 1, 1, 0, 0],
                      [2, 1, 3, 1.5, 0, 0],
                      [1, 1, 1.5, 3, 0, 0],
                      [0, 0, 0, 0, .2, 0.1],
                      [0, 0, 0, 0, 0.1, .2]]) 
    elif scenario == 3:
        S = np.array([[3, 1.5, 1, 1, 0, 0],
                       [1.5, 3, 1, 1, 0, 0],
                       [1, 1, 3, 1.5, 0, 0],
                       [1, 1, 1.5, 3, 0, 0],
                       [0, 0, 0, 0, 1, 0.5],
                       [0, 0, 0, 0, 0.5, 1]])
    elif scenario == 4:
        S = np.array([[6, 3, 2, 1, 0, 0],
                      [3, 6, 1, 1, 0, 0],
                      [2, 1, 3, 1.5, 0, 0],
                      [1, 1, 1.5, 3, 0, 0],
                      [0, 0, 0, 0, 1, 0.5],
                      [0, 0, 0, 0, 0.5, 1]]) 
    elif scenario == 5:
        S = np.array([[3, 1.5, 1, 1, 0, 0],
                       [1.5, 3, 1, 1, 0, 0],
                       [1, 1, 6, 3, 0, 0],
                       [1, 1, 3, 6, 0, 0],
                       [0, 0, 0, 0, .2, 0.1],
                       [0, 0, 0, 0, 0.1, .2]])
    elif scenario == 6:
        S = np.array([[6, 3, 2, 1, 0, 0],
                      [3, 6, 1, 1, 0, 0],
                      [2, 1, 6,  3, 0, 0],
                      [1, 1, 3, 6, 0, 0],
                      [0, 0, 0, 0, .2, 0.1],
                      [0, 0, 0, 0, 0.1, .2]]) 


    filename = "output.txt"
    df, truth = generate_data(n, S)
    ycmoms = get_ycmoms(df, g = g)
    abcmoms = get_abcmoms(df, ycmoms)
    
    alphas = np.array([0, 0.2, 0.3, 0.4, 0.5])
    for a in alphas:
        s = np.sqrt(np.mean((df["X2"] - df["X1"])**2))
        t = s / n ** a if a > 0 else 0
        m = get_medians(df, abcmoms, threshold = t)
        m = pd.concat(m, 1).T
        m["discarded"] = [1, 0]
        m["a"] = a
        m["n"] = n
        m["g"] = g
        m["scenario"] = scenario
        with open(filename, "a") as f:
            m.to_csv(f, header = False, index = False)


    #os.system("qsub job.pbs")

