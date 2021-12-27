import numpy as np

if __name__ == "__main__":
    # read tf output
    tf_res = np.fromfile("../data/pred.bin", dtype=np.float32)
    # read om output
    om_res = np.fromfile("../data/om_pred.bin", dtype=np.float32)

    if np.allclose(tf_res, om_res, 0.001, 0.001) :
        print("[INFO] om output correct")
    else:
        print("[INFO] om output incorrect")