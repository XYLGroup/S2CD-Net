import matlab.engine

def run_matlab_code():
    eng = matlab.engine.start_matlab()

    # Add directory to MATLAB path (required)
    eng.eval("run('/home/descfly/zly/datasets/CAVE/mcodes/generate_test_data.m');", nargout=0)

    # Close engine
    eng.quit()

run_matlab_code()