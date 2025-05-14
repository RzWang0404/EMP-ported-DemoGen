# EMP-ported-DemoGen

The installation and demo execution process is same as [DemoGen](https://github.com/TEA-Lab/DemoGen.git). 

We created and added a `emp_integration.py` file for the `demogen.py` to refer and import EMP model, which replaced the original linear interpolation of the motion segment in `one_stage_augment()` and `two_stage_augment` functions. For the rest, we maintain the original setting as [DemoGen](https://github.com/TEA-Lab/DemoGen.git). 
